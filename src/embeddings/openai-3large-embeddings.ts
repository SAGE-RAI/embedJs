import { OpenAIEmbeddings } from '@langchain/openai';
import { BaseEmbeddings, BaseEmbeddingsParams } from '../interfaces/base-embeddings.js';

export class OpenAi3LargeEmbeddings implements BaseEmbeddings {
    private model: OpenAIEmbeddings;
    private readonly dimension: number;

    constructor(params?: BaseEmbeddingsParams) {
        this.dimension = params?.dimension ?? 3072;

        this.model = new OpenAIEmbeddings({
            modelName: 'text-embedding-3-large',
            maxConcurrency: 3,
            maxRetries: 5,
            apiKey: params?.apiKey ?? undefined,
            dimensions: this.dimension,
        });
    }

    getDimensions(): number {
        return this.dimension;
    }

    embedDocuments(texts: string[]): Promise<number[][]> {
        return this.model.embedDocuments(texts);
    }

    embedQuery(text: string): Promise<number[]> {
        return this.model.embedQuery(text);
    }
}
