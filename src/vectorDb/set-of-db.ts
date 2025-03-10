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
    
            // // Step 4: Calculate the number of chunks to retrieve from each source
            // const chunksPerSource = weights.map(({ db, weight }) => ({
            //     db,
            //     numChunks: Math.round(k * weight) // Round to ensure integer values
            // }));
    
            // // Step 5: Adjust chunk counts to ensure the total is exactly k
            // let totalChunks = chunksPerSource.reduce((sum, { numChunks }) => sum + numChunks, 0);
            // while (totalChunks !== k) {
            //     const diff = k - totalChunks;
            //     const adjustment = diff > 0 ? 1 : -1;
            //     const index = chunksPerSource.findIndex(({ numChunks }) => numChunks + adjustment >= 0);
            //     if (index !== -1) {
            //         chunksPerSource[index].numChunks += adjustment;
            //         totalChunks += adjustment;
            //     }
            // }
    
            // // Step 6: Retrieve chunks from each source based on the calculated number of chunks
            // const results = await Promise.all(chunksPerSource.map(async ({ db, numChunks }) => {
            //     if (numChunks > 0) {
            //         const dbResults = await db.database.similaritySearch(query, numChunks);
            //         return dbResults.map(chunk => ({
            //             ...chunk,
            //             metadata: { ...chunk.metadata, SourceDbName: db.database.constructor.name, dbName: db.name }
            //         }));
            //     }
            //     return [];
            // }));

            // Step 4: Retrieve chunks from the top k sources
            const results = await this.retrieveChunks(weights, query, k);
            return results;
    
            // Step 7: Flatten and return the results
            //return results.flat();
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

    // async similaritySearchTopicClassificationStrategy(query: number[], k: number, rawQuery: string): Promise<ExtractChunkData[]> {
    //     // Step 1: Extract topics and entities from the user's query
    //     try {
    //         // Extract topics and entities from the query using the LLM via silentConversationQuery
    //         const queryExtractionPrompt = `Analyze the following query and extract its primary topics and entities. Assign a weight to each topic/entity based on its importance. Respond with a JSON object: { "topics": { "topic": weight }, "entities": { "entity": weight } }. Do not include any explanations or steps. Query: "${rawQuery}"`;
    //         let queryTopicsAndEntities = { topics: {}, entities: {} };
    //         let queryExtractionResponse = await this.ragApplication.silentConversationQuery(queryExtractionPrompt, null, null, null);
            
    //         try {
    //             queryTopicsAndEntities = JSON.parse(queryExtractionResponse);
    //             if (!queryTopicsAndEntities.topics || !queryTopicsAndEntities.entities) {
    //                 throw new Error('Invalid JSON structure for query topics/entities.');
    //             }
    //         } catch (error) {
    //             console.error("Failed to extract query topics and entities:", error, "Response:", queryExtractionResponse);
    //             queryTopicsAndEntities = { topics: {}, entities: {} }; // Fallback
    //         }

    //         // Step 2: Extract topics and entities from the sources
    //         const sourceTopicEntities = await Promise.all(
    //             this.dbs.map(async db => {
    //                 const fullText = await db.database.getFullText();
    //                 const sourceExtractionPrompt = `Analyze the following text and extract its primary topics and entities. Assign a weight to each topic/entity based on its importance. Respond with a JSON object: { "topics": { "topic": weight }, "entities": { "entity": weight } }. Do not include any explanations or steps. Text: "${fullText}"`;
    //                 let extractionResponse = await this.ragApplication.silentConversationQuery(sourceExtractionPrompt, null, null, null);
    //                 let parsedResponse: { topics: Record<string, number>, entities: Record<string, number> } = { topics: {}, entities: {} };

    //                 try {
    //                     parsedResponse = JSON.parse(extractionResponse);
    //                     if (!parsedResponse.topics || !parsedResponse.entities) {
    //                         throw new Error(`Expected a JSON object with "topics" and "entities" from source ${db.name}.`);
    //                     }
    //                     return { db, topics: parsedResponse.topics, entities: parsedResponse.entities };

    //                 } catch (error) {
    //                     try {
    //                         parsedResponse = JSON.parse(extractionResponse + "}"); // This is a hack to get the error message from the response
    //                         if (!parsedResponse.topics || !parsedResponse.entities) {
    //                             throw new Error(`Hack Failed: Expected a JSON object with "topics" and "entities" from source ${db.name}.`);
    //                         }
    //                         return { db, topics: parsedResponse.topics, entities: parsedResponse.entities }; // Fallback
    //                     } catch (error) {
    //                         console.error("Failed to extract topics for source:", error, "Response:", extractionResponse);
    //                         return { db, topics: {}, entities: {} }; // Fallback
    //                     }
    //                 } 
    //             })
    //         );

    //         // Step 3: Calculate the similarity between the query and each source based on the extracted topics and entities
    //         // Compute relevance scores based on each topic/entity of the full text topics and its intersections to the query topics and entities
    //         const scoredSources = sourceTopicEntities.flat().map(db => {
    //             let relevanceScore = 0;
    //             let totalWeight = 0;
                
    //             // Compute topic relevance
    //             for (const [topic, weight] of Object.entries(db.topics)) {
    //                 if (queryTopicsAndEntities.topics[topic]) {
    //                     let queryWeight = queryTopicsAndEntities.topics[topic];
    //                     relevanceScore += weight * queryWeight;
    //                     totalWeight += queryWeight;
    //                 }
    //             }
                
    //             // Compute entity relevance
    //             for (const [entity, weight] of Object.entries(db.entities)) {
    //                 if (queryTopicsAndEntities.entities[entity]) {
    //                     let queryWeight = queryTopicsAndEntities.entities[entity];
    //                     relevanceScore += weight * queryWeight;
    //                     totalWeight += queryWeight;
    //                 }
    //             }
                
    //             // Normalize score to prevent overemphasis on large counts
    //             if (totalWeight > 0) {
    //                 relevanceScore /= totalWeight;
    //             }
                
    //             return { ...db, relevanceScore };
    //         });
            

    //         // Step 4: Sort sources by relevance score and retrieve chunks from the top k sources
    //         const chunksPerSource = scoredSources.map(({ db, relevanceScore }) => ({
    //             db,
    //             numChunks: Math.round(k * relevanceScore) // Round to ensure integer values
    //         }));
    
    //         // Step 5: Adjust chunk counts to ensure the total is exactly k
    //         let totalChunks = chunksPerSource.reduce((sum, { numChunks }) => sum + numChunks, 0);
    //         while (totalChunks !== k) {
    //             const diff = k - totalChunks;
    //             const adjustment = diff > 0 ? 1 : -1;
    //             const index = chunksPerSource.findIndex(({ numChunks }) => numChunks + adjustment >= 0);
    //             if (index !== -1) {
    //                 chunksPerSource[index].numChunks += adjustment;
    //                 totalChunks += adjustment;
    //             }
    //         }
    
    //         // Step 6: Retrieve chunks from each source based on the calculated number of chunks
    //         const results = await Promise.all(chunksPerSource.map(async ({ db, numChunks }) => {
    //             if (numChunks > 0) {
    //                 const dbResults = await db.database.similaritySearch(query, numChunks);
    //                 return dbResults.map(chunk => ({
    //                     ...chunk,
    //                     metadata: { ...chunk.metadata, SourceDbName: db.database.constructor.name, dbName: db.name }
    //                 }));
    //             }
    //             return [];
    //         }));
    
    //         // Step 7: Flatten and return the results
    //         return results.flat();
            
    //     } catch (error) {
    //         this.debug('Error during topic classification strategy similarity search:', error);
    //         throw error;
    //     }
        
        
    // }

    async similaritySearchTopicClassificationStrategy(query: number[], k: number, rawQuery: string): Promise<ExtractChunkData[]> {
        try {
            // Step 1: Extract topics and entities from the user's query
            const queryTopicsAndEntities = await this.extractTopicsAndEntities(rawQuery);
    
            // Step 2: Extract topics and entities from the sources
            const sourceTopicEntities = await Promise.all(
                this.dbs.map(async db => {
                    const fullText = await db.database.getFullText();
                    // break these full text into  chunks again as the full text is too large for the model tokenizer
                    // Step 1: Split full text into smaller chunks (~512-1024 tokens each)
                    const textChunks = this.chunkText(fullText, 512); // Adjust size as needed
                    
                    // Step 2: Extract topics/entities for each chunk separately
                    const chunkResults = await Promise.all(
                        textChunks.map(async chunk => await this.extractTopicsAndEntities(chunk))
                    );

                    // Step 3: Aggregate extracted topics/entities across all chunks
                    const allTopics = new Set<string>();
                    const allEntities = new Set<string>();
                    chunkResults.forEach(({ topics, entities }) => {
                        Object.keys(topics).forEach(topic => allTopics.add(topic));
                        Object.keys(entities).forEach(entity => allEntities.add(entity));
                    });

                    return { db, topics: Array.from(allTopics), entities: Array.from(allEntities) };
                })
                //     const { topics, entities } = await this.extractTopicsAndEntities(fullText);
                //     return { db, topics, entities };
                // })
            );
    
            // Step 3: Calculate relevance scores
            const scoredSources = sourceTopicEntities.map(({ db, topics, entities }) => {
                const topicsRecord = topics.reduce((acc, topic) => ({ ...acc, [topic]: 1 }), {} as Record<string, number>);
                const entitiesRecord = entities.reduce((acc, entity) => ({ ...acc, [entity]: 1 }), {} as Record<string, number>);
                const relevanceScore = this.computeRelevanceScore(topicsRecord, entitiesRecord, queryTopicsAndEntities);
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
        let response, parsedResponse; 
        response = await this.ragApplication.silentConversationQuery(prompt, null, null, null);    

        try {
            parsedResponse = JSON.parse(response);
            if (!parsedResponse.topics || !parsedResponse.entities) {
                throw new Error('Invalid JSON structure for topics/entities.');
            }
            return parsedResponse;
        } catch (error) {
            try {
                parsedResponse = JSON.parse(response + "}"); // This is a hack to get the error message from the response
                if (!parsedResponse.topics || !parsedResponse.entities) {
                    throw new Error('Invalid JSON structure for topics/entities.');
                }
                return parsedResponse; // Fallback
            } catch (error) {   
                console.error("Failed to extract topics and entities:", error, "Response:", response);
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
                relevanceScore += weight * queryTopicsAndEntities.topics[topic];
                totalWeight += queryTopicsAndEntities.topics[topic];
            }
        }
    
        // Compute entity relevance
        for (const [entity, weight] of Object.entries(sourceEntities)) {
            if (queryTopicsAndEntities.entities[entity]) {
                relevanceScore += weight * queryTopicsAndEntities.entities[entity];
                totalWeight += queryTopicsAndEntities.entities[entity];
            }
        }
    
        // Normalize score
        return totalWeight > 0 ? relevanceScore / totalWeight : 0;
    }
    
    private async retrieveChunks(scoredSources: { db: { database: BaseDb, name: string }, relevanceScore?: number, weight?: number }[], query: number[], k: number): Promise<ExtractChunkData[]> {
        let chunksPerSource;

        // Check if relevanceScore or weight is provided for the sources
        if ('relevanceScore' in scoredSources[0]) {
            // Sort sources by relevance score
            chunksPerSource = scoredSources.map(({ db, relevanceScore }) => ({
                db: db.database,
                numChunks: Math.round(k * relevanceScore)
            }));
        } else if ('weight' in scoredSources[0]) {
            // Sort sources by weight
            chunksPerSource = scoredSources.map(({ db, weight }) => ({
                db: db.database,
                numChunks: Math.round(k * weight)
            }));
        } else {
            // If neither relevanceScore nor weight is present, throw an error or handle the case
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
                    metadata: { ...chunk.metadata, SourceDbName: db.constructor.name, dbName: db.name }
                }));
            }
            return [];
        }));
    
        return results.flat();
    }

    setStrategy(strategy: string): void {
        this.currentStrategy = strategy;
    }

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
    

    async getFullText(): Promise<string> {
        // Fetch and concatenate all stored chunks as full text
        try {
            const allTexts = await Promise.all(this.dbs.map(db => db.database.getFullText()));
            return allTexts.join(' ');
        } catch (error) {
            this.debug('Error fetching full text:', error);
            throw error;
        }
    }
}
