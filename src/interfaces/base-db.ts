import { ExtractChunkData, InsertChunkData } from '../global/types.js';

export interface BaseDb {
    init({}: { dimensions: number }): Promise<void>;
    insertChunks(chunks: InsertChunkData[]): Promise<number>;
    similaritySearch(query: number[], k: number, rawQuery?: string): Promise<ExtractChunkData[]>; // added rawQuery
    getVectorCount(): Promise<number>;

    deleteKeys(uniqueLoaderId: string): Promise<boolean>;
    reset(): Promise<void>;

    getFullText(): Promise<string>; // to get the full text of the database
    getChunks(): Promise<ExtractChunkData[]>; // to get all the chunks in the database
}
