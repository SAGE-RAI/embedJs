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
    private relevanceThreshold: number; // Threshold for selecting sources based on relevance score
    private numberK: number; // Number of chunks to retrieve per source * this overrides the k in the similarity search
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
            case 'topKNChunks':
                return await this.similaritySearchTopKNChunksStrategy(query, this.numberK);
            case 'topKChunksPerSource':
                return await this.similaritySearchTopKChunksPerSourceStrategy(query, this.numberK);
            case 'weightedRelevance':
                return await this.similaritySearchWeightedRelevanceStrategy(query, this.numberK);
            case 'topicClassification':
                return await this.similaritySearchTopicClassificationStrategy(query, this.numberK, rawQuery);
            case 'llmSummarization':
                return await this.similaritySearchLLMSummarization(query, this.numberK, rawQuery);
            default:
                return await this.similaritySearchDefaultStrategy(query, k);    
        }
    }

    async similaritySearchDefaultStrategy(query: number[], k: number): Promise<ExtractChunkData[]> {
        // Fetch results from all databases and attach dbName: Top k chunks across all sources (k chunks)
        const allResults = await Promise.all(
            this.dbs.map(async (db) => {
                const dbResults = await db.database.similaritySearch(query, k);
                return dbResults.map(chunk => ({
                    ...chunk,
                    metadata: { ...chunk.metadata, dbName: db.name } 
                }));
            })
        );
        return allResults.flat();
    }

    async similaritySearchTopKNChunksStrategy(query: number[], k: number): Promise<ExtractChunkData[]> {
        // Top k/n chunks from each of n databases individually (k chunks): The top k chunks are selected from each of the n sources, with each database contributing equally.
        const dbCount = this.dbs.length;
        console.log("dbCount", dbCount);
        if (dbCount === 0) {
            throw new Error('No databases available for similarity search.');
        }
        const chunksPerDb = Math.max(1, Math.ceil(k / dbCount)); // Calculate chunks per data source
        const allResults = await Promise.all(
            this.dbs.map(async (db) => {
                const dbResults = await db.database.similaritySearch(query, chunksPerDb); // in this case we need to change k
                return dbResults.map(chunk => ({
                    ...chunk,
                    metadata: { ...chunk.metadata, dbName: db.name } 
                }));
            })
        );
        return allResults.flat().slice(0, k); // Select top k chunks from all sources
    }

    async similaritySearchTopKChunksPerSourceStrategy(query: number[], k: number): Promise<ExtractChunkData[]> {
        // Top k chunks from each of n sources individually (n*k chunks)
        const allResults = await Promise.all(
            this.dbs.map(async (db) => {
                const dbResults = await db.database.similaritySearch(query, k);  // in this case we need to change k
                // Ensure we select **exactly k** per database
                const topKChunks = dbResults.slice(0, k).map(chunk => ({
                    ...chunk,
                    metadata: { ...chunk.metadata, dbName: db.name }
                }));
                return topKChunks;
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
    
    async similaritySearchTopicClassificationStrategy(query: number[], k: number, rawQuery: string): Promise<ExtractChunkData[]> {
        try {
            // Step 1: Extract topics and entities from the user's query
            const queryTopicsAndEntities = await this.extractTopicsAndEntities(rawQuery);
    
            // Step 2: Extract topics and entities from the sources
            const sourceTopicEntities = await Promise.all(
                this.dbs.map(async db => {
                    const fullText = await db.database.getFullText();
                    // Step 1: Split full text into smaller chunks (~512-1024 tokens each)
                    const textChunks = this.chunkText(fullText, 512); // might reduce to 256
    
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

    async similaritySearchLLMSummarization(query: number[], k: number, rawQuery: string): Promise<ExtractChunkData[]> {
        // // Step 1: Generate full-text embeddings for each source
        // const summarisedSources = await Promise.all(
        //     this.dbs.map(async db => {
        //         const fullText = await db.database.getFullText();
        //         const summarizedText = await this.summarizeText(fullText, rawQuery);
        //         return { db, summarizedText};
        //     })
        // );

        // // Step 2: Combine the summarized results and select the top k chunks
        // const combinedResults = summarisedSources.flatMap(chunk => [
        //     {
        //         ...chunk,
        //         pageContext: chunk.summarizedText.relevanceSummary,
        //         metadata: { dbName: chunk.db.name }
        //     }
        // ]);

        const allResults = await Promise.all(
            this.dbs.map(async (db) => {
                const dbResults = await db.database.similaritySearch(query, k);
                return dbResults.map(chunk => ({
                    ...chunk,
                    metadata: { ...chunk.metadata, dbName: db.name } 
                }));
            })
        );

        // find the summarised text
        const combinedResults = await Promise.all(
            allResults.map(async (chunk) => ({
                ...chunk,
                pageContext: (await this.summarizeText(chunk[0].pageContent, rawQuery, true)),
                metadata: chunk[0].metadata
            }))
        );
        return combinedResults.flat();

        // return [];
    }

    //summarize full text with relevance to the query and in general
    private async summarizeText(text: string, query: string, isRelevant: boolean): Promise<string> {
        try {
            const prompt = isRelevant
            ? `Summarize the following text concisely, focusing on the key points and main ideas relevant to the query: "${query}". Ensure the summary is clear, accurate, and free of opinions or additional information not present in the original text. Here is the text to summarize:\n\n${text}`
            : `Summarize the following text concisely, focusing on the main ideas, key points, and essential details while ensuring clarity and coherence. Do not include personal opinions or any information not present in the original text. Here is the text to summarize:\n\n${text}`;
        
            const summary = await this.ragApplication.silentConversationQuery(prompt, null, 'default', []);
            return summary;
        } catch (error) {
            this.debug('Error during text summarization:', error);
            throw error;
        }
    }

    //summarize full text with relevance to the query and in general
    // private async summarizeText(fullText: string, rawQuery: string): Promise<{relevanceSummary: string, generalSummary: string}> {
    //     if (!this.ragApplication) {
    //         throw new Error('RAGApplication instance is not initialized.');
    //     }
    //     // Step 1a: Summarize with relevance to the query
    //     const promptRelevance = `Summarize the content below, paying attention to what is relevant to the query: "${rawQuery}".  
    //             - Ensure the summary is **concise yet sufficiently detailed** to capture key points.  
    //             - The summary should be **clear, accurate, and contextually relevant**.  
    //             - Avoid opinions, interpretations, or extra information not in the original text.  
    //             Text to summarize: ${fullText}  
    //             Respond with **only** the summary, without extra text or formatting.`;
    //     const responseRelevance = await this.ragApplication.silentConversationQuery(promptRelevance, null, 'default', []);

    //     // Step 2b: Summarize in general (without query focus)
    //     const generalSummaryPrompt = `Summarize the following text clearly and concisely.  
    //     - Capture **main ideas, key points, and essential details**.  
    //     - The summary should be **neutral, structured, and coherent**.  
    //     - **Exclude** personal opinions or any extraneous information.  
    //     Text to summarize: ${fullText}  
    //     Respond with **only** the summary, without extra text or formatting.`;
    //     const responseGeneral = await this.ragApplication.silentConversationQuery(generalSummaryPrompt, null, 'default', []);

    //     return { relevanceSummary: responseRelevance, generalSummary: responseGeneral };
    // }

    private async extractTopicsAndEntities(text: string): Promise<{ topics: Record<string, number>, entities: Record<string, number> }> {
        if (!this.ragApplication) {
            throw new Error('RAGApplication instance is not initialized.');
        }
    
        const prompt = `Analyze the following text and extract at least 5 to 10 primary topics and entities.  
        Each topic and entity should be **distinct** and should capture key themes from the text.  
        Assign a numerical weight (between 0 and 1) to each topic and entity based on its importance.  
        
        Strictly adhere to this response format as a JSON object:
        {
            "topics": { "topic1": weight, "topic2": weight, ..., "topicN": weight },
            "entities": { "entity1": weight, "entity2": weight, ..., "entityN": weight }
        }
        
        - Ensure there are **at least 5 and at most 10** topics.  
        - Ensure there are **at least 5 and at most 10** entities.  
        - Do **not** provide additional explanations or extra text.  
        
        Text to analyze: "${text}"`;
    
        let response: string, topics: Record<string, number>, entities: Record<string, number>;
        response = await this.ragApplication.silentConversationQuery(prompt, null, 'default', []);
        let parsedResponse: { topics: Record<string, number>, entities: Record<string, number> } = { topics: {}, entities: {} };

        try {
            parsedResponse = JSON.parse(response);
            topics = parsedResponse.topics;
            entities = parsedResponse.entities;
            if (!topics || !entities) {
                throw new Error(`Invalid JSON structure for topics/entities.`);
            }
            return { topics, entities };
        } catch (error) {
            try {
                parsedResponse = JSON.parse(response + "}"); // This is a hack to get the error message from the response
                topics = parsedResponse.topics;
                entities = parsedResponse.entities;
                if (!topics || !entities) {
                    throw new Error(`Failed to fix the JSON Parser error`);
                }
                return {topics: topics, entities: entities }; // Fallback
            } catch (error) {
                console.error("Failed to extract topics and entities:", error, "Response:", response);
                return {topics: {}, entities: {} }; // Fallback
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
        let chunksPerSource: { db: { database: BaseDb, name: string }, relevanceScore?:number, numChunks?: number }[] = [];

        // Check if relevanceScore or weight is provided for the sources and calculate the number of chunks to select
        if ('relevanceScore' in scoredSources[0]) {
            chunksPerSource = scoredSources.map(({ db, relevanceScore }) => ({
                db,
                relevanceScore,
            }));
        } else if ('weight' in scoredSources[0]) {
            chunksPerSource = scoredSources.map(({ db, weight }) => ({
                db,
                relevanceScore: weight,
                numChunks: Math.round(k * weight)
            }));

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
        } else {
            throw new Error('Neither relevanceScore nor weight is provided for the sources.');
        }
    
        // Retrieve chunks from each source
        const results = await Promise.all(chunksPerSource.map(async ({ db, relevanceScore, numChunks }) => {
            if (numChunks > 0) { // weightedRelevance
                const dbResults = await db.database.similaritySearch(query, numChunks);
                return dbResults.map(chunk => ({
                    ...chunk,
                    metadata: { ...chunk.metadata, dbName: db.name },
                    weightedRelevance: relevanceScore
                }));
            } else if (relevanceScore >= this.relevanceThreshold) { // topicClassification
                const dbResults = await db.database.similaritySearch(query, k);
                return dbResults.map(chunk => ({
                    ...chunk,
                    metadata: { ...chunk.metadata, dbName: db.name },
                    topicRelevance: relevanceScore
                }));
            }
            return [];
        }));
    
        return results.flat();
    }

    private cosineSimilarity(vecA: number[], vecB: number[]): number {
        const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
        const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
        const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
        return dotProduct / (magnitudeA * magnitudeB);
    }

    // set the strategy for the similarity search
    setStrategy(strategy: string): void {
        this.currentStrategy = strategy;
    }

    // set relevancethreshold
    setRelevanceThreshold(relevanceThreshold: number): void {
        this.relevanceThreshold = relevanceThreshold;
    }

    // set number of chunks to retrieve per source
    setNumberK(numberK: number): void {
        this.numberK = numberK;
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
