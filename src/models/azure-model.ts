import createDebugMessages from 'debug';
import ModelClient from "@azure-rest/ai-inference";
import { isUnexpected } from "@azure-rest/ai-inference";
import { BaseModel } from '../interfaces/base-model.js';
import { Chunk, Message } from '../global/types.js';
import { AzureKeyCredential } from '@azure/core-auth';

export class AzureAIInferenceModel extends BaseModel {
    private readonly debug = createDebugMessages('embedjs:model:AzureAIInference');
    
    private readonly modelName: string;
    private readonly maxNewTokens: number;
    private readonly endpointUrl?: string;
    private readonly apiKey?: string;
    private model: any;

    constructor(params?: { modelName?: string; temperature?: number; maxNewTokens?: number; endpointUrl?: string; apiKey?: string }) {
        super(params?.temperature);

        this.modelName = params?.modelName ?? 'Meta-Llama-3-70B-Instruct';
        this.endpointUrl = params?.endpointUrl;
        this.apiKey = params?.apiKey;
        this.maxNewTokens = params?.maxNewTokens ?? 300;
    }

    override async init(): Promise<void> {
        this.model = ModelClient(
            this.endpointUrl,
            new AzureKeyCredential(this.apiKey));
    }

    override async runQuery(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        pastConversations: Message[],
    ): Promise<string> {
        const pastMessages = [
            {
                role: "system",
                content: system
            }
        ];
        pastMessages.push({
            role: "system",
            content: `Data: ${supportingContext.map((s) => s.pageContent).join('; ')}`
        });

        pastMessages.push.apply(
            pastMessages,
            pastConversations.map((c) => {
                if (c.sender === 'AI') return {
                    role: "assistant",
                    content: c.message
                };
                else if (c.sender === 'SYSTEM') return {
                    role: "system",
                    content: c.message
                };
                else return {
                    role: "user",
                    content: c.message
                };
            }),
        );

        pastMessages.push({
            role: "user",
            content: userQuery
        });

        const finalPrompt = pastMessages//.join('\n');
        // this.debug('Final prompt being sent to Azure - ', finalPrompt);
        this.debug(`Executing Azure AI Inference '${this.modelName}' model with prompt -`, userQuery);
        const response = await this.model.path("chat/completions").post({
             body: {
                model: this.modelName,
                messages: finalPrompt,
                max_tokens: this.maxNewTokens,
                temperature: this.temperature
            }
            });
        if (isUnexpected(response)) {
            throw response.body.error;
        }

        const result = response.body.choices[0].message.content
        this.debug('Azure response -', result);
        return result;
    }
}
