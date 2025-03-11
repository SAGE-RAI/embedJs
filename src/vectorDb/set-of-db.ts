/**
 * author: joseph.kwarteng@open.ac.uk
 * created on: 27-02-2025-18h-35m
 * github: https://github.com/kwartengj
 * copyright 2025
 */
import createDebugMessages from 'debug';
import { BaseDb } from '../interfaces/base-db.js';
import { ExtractChunkData, InsertChunkData } from '../global/types.js';
import { RAGEmbedding } from '../core/rag-embedding.js';
import { RAGApplication } from '../core/rag-application.js';


export class SetOfDbs implements BaseDb {
    private readonly debug = createDebugMessages('embedjs:vector:SetOfDbs');
    private readonly DEFAULT_DB_POSITION = 0;
    private currentStrategy: string;
    private dbs: {
        database: BaseDb,
        name: string
    }[];

    private ragApplication;

    constructor(dbs: {
        database: BaseDb,
        name: string
    }[]) {
        if (!dbs || dbs.length === 0) {
            throw new Error('At least one database must be provided.');
        }
        this.dbs = dbs;
    }

    async init({ dimensions }: { dimensions: number }): Promise<void> {
        this.debug('Connecting to set of database...');
        await Promise.all(this.dbs.map(db => db.database.init({ dimensions })));
        this.debug('Connected to set of database');
    }

    async insertChunks(chunks: InsertChunkData[]): Promise<number> {
        return this.dbs[this.DEFAULT_DB_POSITION].database.insertChunks(chunks);
    }

    async similaritySearch(query: number[], k: number, rawQuery?: string): Promise<ExtractChunkData[]> {
        // apply strategy for similarity search
        switch (this.currentStrategy) {
            case 'weightedRelevance':
                return await this.similaritySearchWeightedRelevanceStrategy(query, k);
            case 'topicClassification':
                return await this.similaritySearchTopicClassificationStrategy(query, k, rawQuery);
            default:
                return await this.similaritySearchDefaultStrategy(query, k);    
        }

    }

    async similaritySearchDefaultStrategy(query: number[], k: number): Promise<ExtractChunkData[]> {
        // Fetch results from all databases and attach dbName
        const allResults = await Promise.all(
            this.dbs.map(async (db) => {
                const dbResults = await db.database.similaritySearch(query, k);
                return dbResults.map(chunk => ({
                    ...chunk,
                    metadata: { ...chunk.metadata, SourceDbName: db.database.constructor.name , dbName: db.name } // Adding dbName
                }));
            })
        );
        return allResults.flat();
    }

    async similaritySearchWeightedRelevanceStrategy(query: number[], k: number): Promise<ExtractChunkData[]> {
        try {
            // Step 1: Generate full-text embeddings for each source
            const embeddings = await Promise.all(
                this.dbs.map(async db => {
                    const fullText = await db.database.getFullText();
                    return { db, embedding: await RAGEmbedding.getEmbedding().embedQuery(fullText) };
                })
            );
    
            // Step 2: Calculate cosine similarity between the query and each source's full-text embedding
            const similarities = embeddings.map(({ db, embedding }) => ({
                db,
                similarity: this.cosineSimilarity(query, embedding)
            }));
    
            // Step 3: Normalize similarities to weights between [0, 1]
            const totalSimilarity = similarities.reduce((sum, { similarity }) => sum + similarity, 0);
            const weights = similarities.map(({ db, similarity }) => ({
                db,
                weight: similarity / totalSimilarity
            }));

            // Step 4: Retrieve chunks from the top k sources
            const results = await this.retrieveChunks(weights, query, k);
            return results;
    
        } catch (error) {
            this.debug('Error during weighted relevance strategy similarity search:', error);
            throw error;
        }
    }

    private cosineSimilarity(vecA: number[], vecB: number[]): number {
        const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
        const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
        const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
        return dotProduct / (magnitudeA * magnitudeB);
    }
    
    async similaritySearchTopicClassificationStrategy(query: number[], k: number, rawQuery: string): Promise<ExtractChunkData[]> {
        try {
            // Step 1: Extract topics and entities from the user's query
            const queryTopicsAndEntities = await this.extractTopicsAndEntities(rawQuery);
    
            // Step 2: Extract topics and entities from the sources
            const sourceTopicEntities = await Promise.all(
                this.dbs.map(async db => {
                    const fullText = await db.database.getFullText();
                    // Step 1: Split full text into smaller chunks (~512-1024 tokens each)
                    const textChunks = this.chunkText(fullText, 512); // Adjust size as needed
    
                    // Step 2: Extract topics/entities for each chunk separately
                    const chunkResults = await Promise.all(
                        textChunks.map(async chunk => await this.extractTopicsAndEntities(chunk))
                    );
    
                    // Step 3: Aggregate extracted topics/entities across all chunks
                    const topicsMap = new Map<string, number>();
                    const entitiesMap = new Map<string, number>();
    
                    chunkResults.forEach(({ topics, entities }) => {
                        for (const [topic, weight] of Object.entries(topics)) {
                            if (topicsMap.has(topic)) {
                                topicsMap.set(topic, topicsMap.get(topic)! + weight); // Accumulate weights
                            } else {
                                topicsMap.set(topic, weight);
                            }
                        }
                        for (const [entity, weight] of Object.entries(entities)) {
                            if (entitiesMap.has(entity)) {
                                entitiesMap.set(entity, entitiesMap.get(entity)! + weight); // Accumulate weights
                            } else {
                                entitiesMap.set(entity, weight);
                            }
                        }
                    });
    
                    return { 
                        db, 
                        topics: Object.fromEntries(topicsMap), // Convert Map to object
                        entities: Object.fromEntries(entitiesMap) // Convert Map to object
                    };
                })
            );
    
            // Step 3: Calculate relevance scores
            const scoredSources = sourceTopicEntities.map(({ db, topics, entities }) => {
                const relevanceScore = this.computeRelevanceScore(topics, entities, queryTopicsAndEntities);
                return { db, relevanceScore };
            });
    
            // Step 4: Retrieve chunks from the top k sources
            const results = await this.retrieveChunks(scoredSources, query, k);
            return results;
        } catch (error) {
            this.debug('Error during topic classification strategy similarity search:', error);
            throw error;
        }
    }

    private async extractTopicsAndEntities(text: string): Promise<{ topics: Record<string, number>, entities: Record<string, number> }> {
        if (!this.ragApplication) {
            throw new Error('RAGApplication instance is not initialized.');
        }
    
        const prompt = `Analyze the following text and extract its primary topics and entities. Assign a weight to each topic/entity based on its importance. Respond with a JSON object: { "topics": { "topic": weight }, "entities": { "entity": weight } }. Do not include any explanations or steps. Text: "${text}"`;
        let response: string, parsedResponse: { topics: Record<string, number>, entities: Record<string, number> } | null = null;
        let chunks = [];
        response = await this.ragApplication.silentConversationQuery(prompt, null, 'default', chunks);
    
        try {
            // Attempt to parse the response as JSON
            parsedResponse = JSON.parse(response);
    
            // Validate the parsed response
            if (!parsedResponse.topics || !parsedResponse.entities) {
                throw new Error('Invalid JSON structure for topics/entities.');
            }
    
            return parsedResponse;
        } catch (error) {
            // If parsing fails, attempt to fix the JSON
            try {
                // Attempt to fix malformed JSON by adding a closing brace if missing
                let fixedResponse = response.trim();
                if (!fixedResponse.endsWith('}')) {
                    fixedResponse += '}';
                }
    
                // Remove trailing commas (if any)
                fixedResponse = fixedResponse.replace(/,\s*}/g, '}');
    
                // Attempt to parse the fixed JSON
                parsedResponse = JSON.parse(fixedResponse);
    
                // Validate the parsed response
                if (!parsedResponse.topics || !parsedResponse.entities) {
                    throw new Error('Invalid JSON structure for topics/entities after fixing.');
                }
    
                return parsedResponse;
            } catch (fixError) {
                // If fixing fails, log the error and return a fallback
                console.error("Failed to extract topics and entities:", fixError, "Response:", response);
                return { topics: {}, entities: {} }; // Fallback
            }
        }
    }

    private computeRelevanceScore(sourceTopics: Record<string, number>, sourceEntities: Record<string, number>, queryTopicsAndEntities: { topics: Record<string, number>, entities: Record<string, number> }): number {
        let relevanceScore = 0;
        let totalWeight = 0;
    
        // Compute topic relevance
        for (const [topic, weight] of Object.entries(sourceTopics)) {
            if (queryTopicsAndEntities.topics[topic]) {
                const queryWeight = queryTopicsAndEntities.topics[topic];
                relevanceScore += weight * queryWeight; // Use float weights
                totalWeight += queryWeight;
            }
        }
    
        // Compute entity relevance
        for (const [entity, weight] of Object.entries(sourceEntities)) {
            if (queryTopicsAndEntities.entities[entity]) {
                const queryWeight = queryTopicsAndEntities.entities[entity];
                relevanceScore += weight * queryWeight; // Use float weights
                totalWeight += queryWeight;
            }
        }
    
        // Normalize score
        return totalWeight > 0 ? relevanceScore / totalWeight : 0;
    }
    
    private async retrieveChunks(scoredSources: { db: { database: BaseDb, name: string }, relevanceScore?: number, weight?: number }[], query: number[], k: number): Promise<ExtractChunkData[]> {
        let chunksPerSource: { db: { database: BaseDb, name: string }, numChunks: number }[] = [];

        // Check if relevanceScore or weight is provided for the sources and calculate the number of chunks to select
        if ('relevanceScore' in scoredSources[0]) {
            chunksPerSource = scoredSources.map(({ db, relevanceScore }) => ({
                db,
                numChunks: Math.round(k * relevanceScore)
            }));
        } else if ('weight' in scoredSources[0]) {
            chunksPerSource = scoredSources.map(({ db, weight }) => ({
                db,
                numChunks: Math.round(k * weight)
            }));
        } else {
            throw new Error('Neither relevanceScore nor weight is provided for the sources.');
        }
    
        // Adjust chunk counts to ensure the total is exactly k
        let totalChunks = chunksPerSource.reduce((sum, { numChunks }) => sum + numChunks, 0);
        while (totalChunks !== k) {
            const diff = k - totalChunks;
            const adjustment = diff > 0 ? 1 : -1;
            const index = chunksPerSource.findIndex(({ numChunks }) => numChunks + adjustment >= 0);
            if (index !== -1) {
                chunksPerSource[index].numChunks += adjustment;
                totalChunks += adjustment;
            }
        }
    
        // Retrieve chunks from each source
        const results = await Promise.all(chunksPerSource.map(async ({ db, numChunks }) => {
            if (numChunks > 0) {
                const dbResults = await db.database.similaritySearch(query, numChunks);
                return dbResults.map(chunk => ({
                    ...chunk,
                    metadata: { ...chunk.metadata, sourceDbName: db.constructor.name, dbName: db.name }
                }));
            }
            return [];
        }));
    
        return results.flat();
    }

    // set the strategy for the similarity search
    setStrategy(strategy: string): void {
        this.currentStrategy = strategy;
    }

    // set the ragApp initialised in the main app
    ragApp(ragApp: RAGApplication): void {
        this.ragApplication = ragApp;
    }

    async getVectorCount(): Promise<number> {
        const counts = await Promise.all(this.dbs.map(db => db.database.getVectorCount()));
        return counts.reduce((sum, count) => sum + count, 0);
    }

    async deleteKeys(uniqueLoaderId: string): Promise<boolean> {
        const results = await Promise.all(this.dbs.map(db => db.database.deleteKeys(uniqueLoaderId)));
        return results.every(result => result);
    }

    async reset(): Promise<void> {
        await Promise.all(this.dbs.map(db => db.database.reset()));
    }

    // chunk the fulltext into smaller chunks
    chunkText(text: string, maxTokens: number): string[] {
        const words = text.split(/\s+/); // Split by spaces (or use a tokenizer for better control)
        const chunks = [];
        
        for (let i = 0; i < words.length; i += maxTokens) {
            chunks.push(words.slice(i, i + maxTokens).join(' '));
        }
        return chunks;
    }
    
    // Fetch and concatenate all stored chunks as full text
    async getFullText(): Promise<string> {
        try {
            const allTexts = await Promise.all(this.dbs.map(db => db.database.getFullText()));
            return allTexts.join(' ');
        } catch (error) {
            this.debug('Error fetching full text:', error);
            throw error;
        }
    }
}
