/**
 * author: joseph.kwarteng@open.ac.uk
 * created on: 02-07-2024-23h-33m
 * github: https://github.com/kwartengj
 * copyright 2024
*/

import { OpenAIEmbeddings } from '@langchain/openai';
import { BaseEmbeddings } from '../interfaces/base-embeddings.js';

export class OpenAiGenericEmbeddings implements BaseEmbeddings {
    private model: OpenAIEmbeddings;
    private readonly dimensions: number;

    constructor({ modelName, baseURL, dimensions} : { modelName: string, baseURL : string, dimensions: number}) {
        this.model = new OpenAIEmbeddings({ modelName: modelName, maxConcurrency: 3, maxRetries: 5 }, { baseURL: baseURL});
        this.dimensions = dimensions;
    }

    getDimensions(): number {
        return this.dimensions;
    }

    embedDocuments(texts: string[]): Promise<number[][]> {
        return this.model.embedDocuments(texts);
    }

    embedQuery(text: string): Promise<number[]> {
        return this.model.embedQuery(text);
    }
}
