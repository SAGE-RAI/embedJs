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

export class SetOfDbs implements BaseDb {
    private readonly debug = createDebugMessages('embedjs:vector:SetOfDbs');
    private readonly DEFAULT_DB_POSITION = 0;
    private currentStrategy: string;
    private dbs: {
        database: BaseDb,
        name: string
    }[];

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

    async similaritySearch(query: number[], k: number): Promise<ExtractChunkData[]> {
        // apply strategy for similarity search
        switch (this.currentStrategy) {
            case 'weightedRelevance':
                return await this.similaritySearchWeightedRelevanceStrategy(query, k);
            case 'topicClassification':
                throw new Error('Topic classification strategy not implemented yet');
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

                // this.dbs.map(async db => {
                //     const chunks = await db.database.similaritySearch([], 100);
                //     const fullText = chunks.map(chunk => chunk.metadata?.originalText || chunk.pageContent).join(' ');
                //     return { db, embedding: await RAGEmbedding.getEmbedding().embedQuery(fullText) };
                // })
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
    
            // Step 4: Calculate the number of chunks to retrieve from each source
            const chunksPerSource = weights.map(({ db, weight }) => ({
                db,
                numChunks: Math.round(k * weight) // Round to ensure integer values
            }));
    
            // Step 5: Adjust chunk counts to ensure the total is exactly k
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
    
            // Step 6: Retrieve chunks from each source based on the calculated number of chunks
            const results = await Promise.all(chunksPerSource.map(async ({ db, numChunks }) => {
                if (numChunks > 0) {
                    const dbResults = await db.database.similaritySearch(query, numChunks);
                    return dbResults.map(chunk => ({
                        ...chunk,
                        metadata: { ...chunk.metadata, SourceDbName: db.database.constructor.name, dbName: db.name }
                    }));
                }
                return [];
            }));
    
            // Step 7: Flatten and return the results
            return results.flat();
        } catch (error) {
            this.debug('Error during weighted relevance strategy similarity search:', error);
            throw error;
        }
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

    private cosineSimilarity(vecA: number[], vecB: number[]): number {
        const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
        const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
        const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
        return dotProduct / (magnitudeA * magnitudeB);
    }

    // async similaritySearchTopicClassificationStrategy(query: number[], k: number): ExtractChunkData[] | PromiseLike<ExtractChunkData[]> {
    //     // ...do topic classification over the dbs
    //     throw new Error('Method not implemented.');
    
    // }

    setStrategy(strategy: string): void {
        this.currentStrategy = strategy;
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
}
