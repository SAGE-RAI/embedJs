import createDebugMessages from 'debug';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, AIMessage, SystemMessage } from '@langchain/core/messages';

import { BaseGenerationParams, BaseModel } from '../interfaces/base-model.js';
import { Chunk, Message } from '../global/types.js';

export class OpenAi extends BaseModel {
    private readonly debug = createDebugMessages('embedjs:model:OpenAi');
    private readonly modelName: string;
    private readonly baseURL: string;
    private readonly apiKey?: string;
    private model: ChatOpenAI;

    constructor(params? : BaseGenerationParams) {
        super(params);
        this.modelName = params.modelName;
        if (params.baseUrl !== undefined) {
            this.baseURL = params.baseUrl;
        }
        if (params.apiKey !== undefined) {
            this.apiKey = params.apiKey
        }
    }

    override async init(): Promise<void> {
        this.model = new ChatOpenAI({ temperature: this.temperature, model: this.modelName, apiKey: this.apiKey },
            this.baseURL ? { baseURL: this.baseURL } : {});
    }

    override async runQuery(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        pastConversations: Message[],
    ): Promise<string> {
        const pastMessages: (AIMessage | SystemMessage | HumanMessage)[] = [new SystemMessage(system)];
        pastMessages.push(
            new SystemMessage(`Supporting context: ${supportingContext.map((s) => s.pageContent).join('; ')}`),
        );

        pastMessages.push.apply(
            pastMessages,
            pastConversations.map((c) => {
                if (c.sender === 'AI') return new AIMessage({ content: c.message });
                else if (c.sender === 'SYSTEM') return new SystemMessage({ content: c.message });
                else return new HumanMessage({ content: c.message });
            }),
        );
        pastMessages.push(new HumanMessage(`${userQuery}?`));

        this.debug('Executing openai model with prompt -', userQuery);
        const result = await this.model.invoke(pastMessages);
        this.debug('OpenAI response -', result);
        return result.content.toString();
    }
}