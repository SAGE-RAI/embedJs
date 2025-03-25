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
        k = this.numberK; // override the k
        switch (this.currentStrategy) {
            case 'topKNChunks':
                return await this.similaritySearchTopKNChunksStrategy(query, k);
            case 'topKChunksPerSource':
                return await this.similaritySearchTopKChunksPerSourceStrategy(query, k);
            case 'weightedRelevance':
                return await this.similaritySearchWeightedRelevanceStrategy(query, k);
            case 'topicClassification':
                return await this.similaritySearchTopicClassificationStrategy(query, k, rawQuery);
            case 'llmSummarization':
                return await this.similaritySearchLLMSummarization(query, k, rawQuery);
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
        const chunksPerDb = Math.ceil(k / this.dbs.length); // Calculate chunks per data source
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
                similarity: this.euclideanDistance(this.normalize(query), this.normalize(embedding))
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
    

    // async similaritySearchTopicClassificationStrategy(query: number[], k: number, rawQuery: string): Promise<ExtractChunkData[]> {
    //     try {
    //         // Step 1: Extract topics and entities from the user's query
    //         const queryTopicsAndEntities = await this.extractTopicsAndEntities(rawQuery);

    //         // Step 2: Extract topics and entities from the sources
    //         const sourceTopicEntities = await Promise.all(
    //             this.dbs.map(async db => {
    //                 const fullText = await db.database.getFullText();
    //                 // Step 1: Split full text into smaller chunks (~512-1024 tokens each)
    //                 const textChunks = this.chunkText(fullText, 512); // might reduce to 256

    //                 // Step 2: Extract topics/entities for each chunk separately
    //                 const chunkResults = await Promise.all(
    //                     textChunks.map(async chunk => await this.extractTopicsAndEntities(chunk))
    //                 );

    //                 // Step 3: Aggregate extracted topics/entities across all chunks
    //                 const topicsMap = new Map<string, number>();
    //                 const entitiesMap = new Map<string, number>();

    //                 chunkResults.forEach(({ topics, entities }) => {
    //                     for (const [topic, weight] of Object.entries(topics)) {
    //                         if (topicsMap.has(topic)) {
    //                             topicsMap.set(topic, topicsMap.get(topic)! + weight); // Accumulate weights
    //                         } else {
    //                             topicsMap.set(topic, weight);
    //                         }
    //                     }
    //                     for (const [entity, weight] of Object.entries(entities)) {
    //                         if (entitiesMap.has(entity)) {
    //                             entitiesMap.set(entity, entitiesMap.get(entity)! + weight); // Accumulate weights
    //                         } else {
    //                             entitiesMap.set(entity, weight);
    //                         }
    //                     }
    //                 });

    //                 return { 
    //                     db, 
    //                     topics: Object.fromEntries(topicsMap), // Convert Map to object
    //                     entities: Object.fromEntries(entitiesMap) // Convert Map to object
    //                 };
    //             })
    //         );

    //         // Step 3: Calculate relevance scores
    //         const scoredSources = sourceTopicEntities.map(({ db, topics, entities }) => {
    //             const relevanceScore = this.computeRelevanceScore(topics, entities, queryTopicsAndEntities);
    //             return { db, relevanceScore };
    //         });

    //         // Step 4: Retrieve chunks from the top k sources
    //         const results = await this.retrieveChunks(scoredSources, query, k);
    //         return results;
    //     } catch (error) {
    //         this.debug('Error during topic classification strategy similarity search:', error);
    //         throw error;
    //     }
    // }


    // async similaritySearchTopicClassificationStrategy(query: number[], k: number, rawQuery: string): Promise<ExtractChunkData[]> {
    //     try {
    //         // Step 1: Extract topics and entities from the user's query
    //         const queryTopicsAndEntities = await this.extractTopicsAndEntities(rawQuery);
    
    //         // Step 2: Extract topics and entities from the sources
    //         const sourceTopicEntities = await Promise.all(
    //             this.dbs.map(async db => {
    //                 // Get the chunks from the database
    //                 const chunks = await db.database.getChunks();
    
    //                 // Process chunks in parallel with a concurrency limit
    //                 const chunkResults = await this.processChunksWithConcurrency(chunks, async chunk => {
    //                     // Check if topics and entities are already cached in the chunk metadata
    //                     if (chunk.metadata.topics && chunk.metadata.entities) {
    //                         return {
    //                             topics: typeof chunk.metadata.topics === 'string' ? JSON.parse(chunk.metadata.topics) : {},
    //                             entities: typeof chunk.metadata.entities === 'string' ? JSON.parse(chunk.metadata.entities) : {}
    //                         };
    //                     }
    
    //                     // Extract topics and entities from the chunk
    //                     const { topics, entities } = await this.extractTopicsAndEntities(chunk.pageContent);
    
    //                     // Cache the topics and entities in the chunk metadata for future use
    //                     chunk.metadata.topics = JSON.stringify(topics);
    //                     chunk.metadata.entities = JSON.stringify(entities);
    
    //                     // Update the chunk metadata in the database (optional, can be done asynchronously)
    //                     // await db.database.updateChunkMetadata(chunk.id, chunk.metadata);
    
    //                     return { topics, entities };
    //                 });
    
    //                 // Step 3: Aggregate extracted topics/entities across all chunks
    //                 const topicsMap = new Map<string, number>();
    //                 const entitiesMap = new Map<string, number>();
    
    //                 chunkResults.forEach(({ topics, entities }) => {
    //                     for (const [topic, weight] of Object.entries(topics)) {
    //                         topicsMap.set(topic, (topicsMap.get(topic) || 0) + (weight as number));
    //                     }
    //                     for (const [entity, weight] of Object.entries(entities)) {
    //                         entitiesMap.set(entity, (entitiesMap.get(entity) || 0) + (weight as number));
    //                     }
    //                 });
    
    //                 return {
    //                     db,
    //                     topics: Object.fromEntries(topicsMap), // Convert Map to object
    //                     entities: Object.fromEntries(entitiesMap) // Convert Map to object
    //                 };
    //             })
    //         );
    
    //         // Step 3: Calculate relevance scores
    //         const scoredSources = sourceTopicEntities.map(({ db, topics, entities }) => {
    //             const relevanceScore = this.computeRelevanceScore(topics, entities, queryTopicsAndEntities);
    //             return { db, relevanceScore };
    //         });
    
    //         // Step 4: Retrieve chunks from the top k sources
    //         const results = await this.retrieveChunks(scoredSources, query, k);
    //         return results;
    //     } catch (error) {
    //         this.debug('Error during topic classification strategy similarity search:', error);
    //         throw error;
    //     }
    // }

    // new implementation of the topic classification strategy
    async similaritySearchTopicClassificationStrategy(query: number[], k: number, rawQuery: string): Promise<ExtractChunkData[]> {
        try {
            // Step 1: Extract topics and entities from the user's query
            const queryTopicsAndEntities = await this.extractTopicsAndEntities(rawQuery);
    
            // Step 2: Process chunks in batches to avoid limit errors
            const batchSize = 100; // Adjust based on system limits
            const sourceTopicEntities = await Promise.all(
                this.dbs.map(async db => {
                    const chunks = await db.database.getChunks();
                    const topicsMap = new Map<string, number>();
                    const entitiesMap = new Map<string, number>();
    
                    // Process chunks in batches
                    for (let i = 0; i < chunks.length; i += batchSize) {
                        const batch = chunks.slice(i, i + batchSize);
                        const batchResults = await Promise.all(
                            batch.map(async chunk => {
                                // Check if topics and entities are already cached in metadata
                                if (chunk.metadata?.topics && chunk.metadata?.entities) {
                                    return {
                                        topics: JSON.parse(chunk.metadata.topics as string),
                                        entities: JSON.parse(chunk.metadata.entities as string)
                                    };
                                } else {
                                    // Extract topics and entities if not cached
                                    const result = await this.extractTopicsAndEntities(chunk.pageContent);

                                    // Cache the result in the chunk metadata
                                    chunk.metadata = {
                                        ...chunk.metadata,
                                        topics: JSON.stringify(result.topics),
                                        entities: JSON.stringify(result.entities)
                                    };
                                    return result;
                                }
                            })
                        );

                        // Aggregate results for the current batch
                        batchResults.forEach(({ topics, entities }) => {
                            for (const [topic, weight] of Object.entries(topics)) {
                                topicsMap.set(topic, (topicsMap.get(topic) || 0) + (weight as number));
                            }
                            for (const [entity, weight] of Object.entries(entities)) {
                                entitiesMap.set(entity, (entitiesMap.get(entity) || 0) + (weight as number));
                            }
                        });
                    }
    
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

    // async similaritySearchLLMSummarization(query: number[], k: number, rawQuery: string): Promise<ExtractChunkData[]> {
    //     // Step 1: Generate full-text embeddings for each source
    //     const summarisedSources = await Promise.all(
    //         this.dbs.map(async db => {
    //             const fullText = await db.database.getFullText();
    //             const summarizedText = await this.summarizeText(fullText, rawQuery, true);
    //             const embeddingRelevance = await RAGEmbedding.getEmbedding().embedQuery(summarizedText);
    //             return { db, embeddingRelevance };
    //         })
    //     );

    //     // Step 2: Calculate euclidean similarity between the query and each source's full-text embedding
    //     const similarities = summarisedSources.map(({ db, embeddingRelevance }) => ({
    //         db,
    //         similarity: this.euclideanDistance(this.normalize(query), this.normalize(embeddingRelevance))
    //     }));

    //     // Step 3: Check the similarity score and select the top k sources
    //     const selectedSource = similarities.reduce((highest, current) => {
    //         return current.similarity > highest.similarity ? current : highest;
    //     }, similarities[0]);

    //     // Step 4: Retrieve chunks from the db with the highest similarity score
    //     const results = await selectedSource.db.database.similaritySearch(query, k);
    //     return results.map(chunk => ({
    //         ...chunk,
    //         metadata: { ...chunk.metadata, dbName: selectedSource.db.name }
    //     }));
    // }

    //summarize full text with relevance to the query and in general
    
    async similaritySearchLLMSummarization(query: number[], k: number, rawQuery: string): Promise<ExtractChunkData[]> {
        try {
            // Step 1: Summarize and embed each source in chunks to handle token limits
            // const summarisedSources = await Promise.all(
            //     this.dbs.map(async db => {
            //         const fullText = await db.database.getFullText();
    
            //         // Step 1a: Split full text into smaller chunks to avoid token limits
            //         const textChunks = this.chunkText(fullText, 1024); // Adjust chunk size as needed
    
            //         // Step 1b: Summarize each chunk and combine the summaries
            //         const chunkSummaries = await Promise.all(
            //             textChunks.map(async chunk => await this.summarizeText(chunk, rawQuery, true))
            //         );
            //         const combinedSummary = chunkSummaries.join(" ");
    
            //         // Step 1c: Generate embeddings for the combined summary
            //         const embeddingRelevance = await RAGEmbedding.getEmbedding().embedQuery(combinedSummary);
            //         return { db, embeddingRelevance };
            //     })
            // );

            // Step 1: Process each database using its existing chunks
            const batchSize = 100; // Adjust based on system limits
            const summarisedSources = await Promise.all(
                this.dbs.map(async (db) => {
                    // Get pre-chunked content directly
                    const chunks = await db.database.getChunks();

                    // Step 1a: Summarize each chunk and combine
                    const chunkSummaries: string[] = [];
                    for (let i = 0; i < chunks.length; i += batchSize) {
                        const batch = chunks.slice(i, i + batchSize);
                        const batchSummaries = await Promise.all(
                            batch.map(async (chunk) => {
                                try {
                                    // Summarize the chunk
                                    return await this.summarizeText(chunk.pageContent, rawQuery, true);
                                } catch (error) {
                                    this.debug('Error summarizing chunk:', error);
                                    return ''; // Fallback to an empty summary for failed chunks
                                }
                            })
                        );
                        chunkSummaries.push(...batchSummaries.filter(summary => summary.trim() !== '')); // Filter out empty summaries
                    }

                    // Combine summaries with context
                    const combinedSummary = chunkSummaries.join('\n\n');

                    // Step 1b: Generate embedding for combined summary
                    let embeddingRelevance: number[];
                    try {
                        embeddingRelevance = await RAGEmbedding.getEmbedding().embedQuery(combinedSummary);
                    } catch (error) {
                        this.debug('Error generating embedding for combined summary:', error);
                        embeddingRelevance = []; // Fallback to an empty embedding
                    }

                    return { db, embeddingRelevance };
                })
            );
            // Step 2: Calculate Euclidean similarity between the query and each source's embedding
            const normalizedQuery = this.normalize(query);
            const similarities = summarisedSources.map(({ db, embeddingRelevance }) => ({
                db,
                similarity: this.euclideanDistance(normalizedQuery, this.normalize(embeddingRelevance))
            }));
    
            // Step 3: Select the source with the highest similarity score
            const selectedSource = similarities.reduce((highest, current) => {
                return current.similarity > highest.similarity ? current : highest;
            }, similarities[0]);
    
            // Step 4: Retrieve chunks from the db with the highest similarity score
            const results = await selectedSource.db.database.similaritySearch(query, k);
            return results.map(chunk => ({
                ...chunk,
                metadata: { ...chunk.metadata, dbName: selectedSource.db.name }
            }));
        } catch (error) {
            this.debug('Error during LLM summarization similarity search:', error);
            throw error;
        }
    }

    // async similaritySearchLLMSummarization(query: number[], k: number, rawQuery: string): Promise<ExtractChunkData[]> {
    //     try {
    //         // Step 1: Process each database using its existing chunks
    //         const summarisedSources = await Promise.all(
    //             this.dbs.map(async (db) => {
    //                 // Get pre-chunked content directly
    //                 const chunks = await db.database.getChunks();
                    
    //                 // Step 1a: Summarize each chunk and combine
    //                 const chunkSummaries = await Promise.all(
    //                     chunks.map(async (chunk) => {
    //                         // Otherwise generate and cache summary
    //                         const summary = await this.summarizeText(chunk.pageContent, rawQuery, true);
    //                         return summary;
    //                     })
    //                 );
                    
    //                 // Combine summaries with context
    //                 const combinedSummary = chunkSummaries.join("\n\n");
                    
    //                 // Step 1b: Generate embedding for combined summary
    //                 const embeddingRelevance = await RAGEmbedding.getEmbedding().embedQuery(combinedSummary);
                    
    //                 return { db, embeddingRelevance, chunks };
    //             })
    //         );
    
    //         // Step 2: Calculate composite scores
    //         const normalizedQuery = this.normalize(query);
            
    //         const scoredSources = summarisedSources.map(({db, embeddingRelevance}) => {
    //             const similarity = this.euclideanDistance(normalizedQuery, this.normalize(embeddingRelevance));
    //             const confidence = 1 / (1 + Math.exp(-similarity));
    //             return { db, similarity, confidence };
    //         });
    
    //         // Step 3: Select best source
    //         const selectedSource = scoredSources.reduce((best, current) => 
    //             (current.similarity * current.confidence) > (best.similarity * best.confidence) 
    //                 ? current 
    //                 : best
    //         );
    
    //         // Step 4: Retrieve chunks from selected source (already available)
    //         const sourceData = summarisedSources.find(s => s.db.name === selectedSource.db.name);
    //         const results = sourceData?.chunks
    //             .sort((a, b) => b.metadata?.relevanceScore - a.metadata?.relevanceScore)
    //             .slice(0, k)
    //             .map(chunk => ({
    //                 ...chunk,
    //                 metadata: {
    //                     ...chunk.metadata,
    //                     dbName: selectedSource.db.name,
    //                     selectionScore: selectedSource.similarity * selectedSource.confidence
    //                 }
    //             })) || [];
    
    //         return results;
    //     } catch (error) {
    //         this.debug('Error during optimized similarity search:', error);
    //         throw error;
    //     }
    // }


    private async summarizeText(text: string, query: string, isRelevant: boolean): Promise<string> {
        try {
            const prompt = isRelevant
            ? `Summarize the following content, focusing on the key points and main ideas relevant to the query: "${query}". Ensure the summary is clear, accurate, and free of opinions or additional information not present in the original text. Do not include personal opinions or any information not present in the original text. Here is the text to summarize:\n\n${text}`
            : `Summarize the following content, focusing on the main ideas, key points, and essential details while ensuring clarity and coherence. Do not include personal opinions or any information not present in the original text. Here is the text to summarize:\n\n${text}`;
        
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
    
        const prompt = `Analyze the text below and extract **3 to 7 distinct primary topics and entities**.  
        Assign a weight (0 to 1) to each based on importance.  
        Respond **strictly** in this JSON format:  
        {  
            "topics": { "topic1": weight, "topic2": weight, ..., "topicN": weight },  
            "entities": { "entity1": weight, "entity2": weight, ..., "entityN": weight }  
        }  
        
        **Rules**:  
        1. Extract **3 to 7 topics** and **3 to 7 entities**.  
        2. Weights must reflect importance (higher = more important).  
        3. Ensure all topics and entities are **distinct and relevant**.  
        4. Do **not** include explanations or extra text.  

        Text: "${text}"`;
    
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
        // function to compute relevance for a given set of items
        const computeRelevance = (
            sourceItems: Record<string, number>,
            queryItems: Record<string, number>
        ): { score: number, totalWeight: number } => {
            let score = 0;
            let totalWeight = 0;

            for (const [item, weight] of Object.entries(sourceItems)) {
                if (queryItems[item]) {
                    const queryWeight = queryItems[item];
                    score += weight * queryWeight; // Use float weights
                    totalWeight += queryWeight;
                }
            }

            return { score, totalWeight };
        };

        // Compute relevance for topics and entities
        const topicRelevance = computeRelevance(sourceTopics, queryTopicsAndEntities.topics);
        const entityRelevance = computeRelevance(sourceEntities, queryTopicsAndEntities.entities);

        // Combine scores and weights
        const totalScore = topicRelevance.score + entityRelevance.score;
        const totalWeight = topicRelevance.totalWeight + entityRelevance.totalWeight;

        // Normalize score and ensure it's between 0 and 1
        return totalWeight > 0 ? Math.min(1, totalScore / totalWeight) : 0;
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

    // switch to the ecludian distance for the similarity search **
    private cosineSimilarity(vecA: number[], vecB: number[]): number {
        const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
        const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
        const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
        return dotProduct / (magnitudeA * magnitudeB);
    }

    // normalize a vector
    private normalize(vec: number[]): number[] {
        const magnitude = Math.sqrt(vec.reduce((sum, val) => sum + val ** 2, 0));
        return vec.map(val => val / magnitude);
    }

    // calculate Euclidean distance
    private euclideanDistance(vecA: number[], vecB: number[]): number{
        const cosSim = this.cosineSimilarity(vecA, vecB);  // Calculate cosine similarity
        return Math.sqrt(2 * (1 - cosSim));
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

    async getChunks(): Promise<ExtractChunkData[]> {
        // Returning getting all chunks from all sources
        const allChunks = await Promise.all(
            this.dbs.map(async db => db.database.getChunks())
        );
        return allChunks.flat();
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
}
