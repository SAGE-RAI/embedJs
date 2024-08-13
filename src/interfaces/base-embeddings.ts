import { BaseModelParams } from "./base-model.js";

export interface BaseEmbeddings {
    embedDocuments(texts: string[]): Promise<number[][]>;
    embedQuery(text: string): Promise<number[]>;
    getDimensions(): number;
}

export interface BaseEmbeddingsParams extends BaseModelParams {
    dimension? : number
}