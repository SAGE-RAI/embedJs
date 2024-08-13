import { OpenAIEmbeddings } from '@langchain/openai';
import { BaseEmbeddings } from '../interfaces/base-embeddings.js';

export class AdaEmbeddings implements BaseEmbeddings {
    private model: OpenAIEmbeddings;

    constructor( params? : { apiKey?: string}) {
        this.model = new OpenAIEmbeddings({ modelName: 'text-embedding-ada-002', maxConcurrency: 3, maxRetries: 5, apiKey: params?.apiKey ?? undefined });
    }

    getDimensions(): number {
        return 1536;
    }

    embedDocuments(texts: string[]): Promise<number[][]> {
        return this.model.embedDocuments(texts);
    }

    embedQuery(text: string): Promise<number[]> {
        return this.model.embedQuery(text);
    }
}
