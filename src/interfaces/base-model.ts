import createDebugMessages from 'debug';
import { v4 as uuidv4 } from 'uuid';
import { Chunk, Message, ConversationEntry, Sources } from '../global/types.js';
import { BaseConversations } from './base-conversations.js';

export abstract class BaseModel {
    private static defaultTemperature: number;
    private static conversations: BaseConversations; // Static property for managing conversations

    private readonly baseDebug = createDebugMessages('embedjs:model:BaseModel');
    private readonly _temperature?: number;

    constructor(temperature?: number) {
        this._temperature = temperature;
    }

    public static setDefaultTemperature(temperature?: number) {
        BaseModel.defaultTemperature = temperature;
    }

    public static setConversations(conversations: BaseConversations) {
        BaseModel.conversations = conversations; // Correct setting of the static property
    }

    public get temperature() {
        return this._temperature ?? BaseModel.defaultTemperature;
    }

    public async init(): Promise<void> {}

    public async query(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        conversationId: string = 'default',
    ): Promise<any> {
        const conversation = await BaseModel.conversations.getConversation(conversationId); // Use static property

        this.baseDebug(`${conversation.entries.length} history entries found for conversationId '${conversationId}'`);

        const uniqueSources = this.extractUniqueSources(supportingContext);

        const chunksIds = this.extractChunkIds(supportingContext);

        // Extract only the content from each entry in the conversation
        const pastConversations = conversation.entries.map(entry => entry.content);

        const result = await this.runQuery(system, userQuery, supportingContext, pastConversations);

        // Add user query to history
        await BaseModel.conversations.addEntryToConversation(conversationId, {
            _id: uuidv4(),
            timestamp: new Date(),
            content: {
                sender: 'HUMAN',
                message: userQuery
            },
            sources: [],
            chunks: []
        });

        const newEntry: ConversationEntry = {
            _id: uuidv4(),
            timestamp: new Date(),
            content: {
                sender: "AI",
                message: result
            },
            sources: uniqueSources,
            chunks: chunksIds
        }
        // Add AI response to history
        await BaseModel.conversations.addEntryToConversation(conversationId, newEntry);

        return newEntry;
    }

    /*
     * silentConversationQuery
     * Do a query based upon the conversation history, but do not add the query or the response to the conversation.
     * This can be used to ask the LLM about the conversation, e.g. if the topic has changed or newer chunks are required etc.
     */
    public async silentConversationQuery(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        conversationId: string = 'default',
    ): Promise<any> {
        const conversation = await BaseModel.conversations.getConversation(conversationId); // Use static property

        this.baseDebug(`${conversation.entries.length} history entries found for conversationId '${conversationId}'`);

        // Extract only the content from each entry in the conversation
        const pastConversations = conversation.entries.map(entry => entry.content);

        const result = await this.runQuery(system, userQuery, supportingContext, pastConversations);

        return result;
    }

    private extractUniqueSources(supportingContext: Chunk[]): Sources[] {
        const uniqueSources = new Map<string, Sources>();  // Use a Map to track unique sources by URL

        supportingContext.forEach(item => {
            const { metadata } = item;
            if (metadata && metadata.source) {
                // Use the source URL as the key to ensure uniqueness
                if (!uniqueSources.has(metadata.source)) {
                    uniqueSources.set(metadata.source, {
                        source: metadata.source,
                        loaderId: metadata.uniqueLoaderId // Assuming this field always exists
                    });
                }
            }
        });

        // Convert the values of the Map to an array
        return Array.from(uniqueSources.values());
    }

    private extractChunkIds(supportingContext: Chunk[]): string[] {
        // Create a Set to track unique chunk IDs
        const chunkIds = new Set<string>();

        supportingContext.forEach(item => {
            const { metadata } = item;
            if (metadata && metadata.id) {
                // Add the chunk ID to the Set (automatically handles uniqueness)
                chunkIds.add(metadata.id);
            }
        });

        // Convert the Set to an array
        return Array.from(chunkIds);
    }

    protected abstract runQuery(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        pastConversations: Message[],
    ): Promise<any>;
}