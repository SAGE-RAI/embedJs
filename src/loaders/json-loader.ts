import md5 from 'md5';

import { BaseLoader } from '../interfaces/base-loader.js';
import { cleanString, truncateCenterString } from '../util/strings.js';

export class JsonLoader extends BaseLoader<{ type: 'JsonLoader' }> {
    private readonly object: Record<string, unknown> | Record<string, unknown>[];
    private readonly pickKeysForEmbedding?: string[];
    private readonly recurse?: boolean;

    constructor({
        object,
        pickKeysForEmbedding,
        recurse
    }: {
        object: Record<string, unknown> | Record<string, unknown>[];
        pickKeysForEmbedding?: string[];
        recurse?: boolean;
    }) {
        super(`JsonLoader_${md5(cleanString(JSON.stringify(object)))}`, {
            object: truncateCenterString(JSON.stringify(object), 50),
        });

        this.pickKeysForEmbedding = pickKeysForEmbedding;
        this.object = object;
        this.recurse = recurse;
    }

    async *recursivelyChunk(obj, objIdentifier, currentPath) {
        for (let key in obj) {
            let nextPathSegment: string;
            if (Array.isArray(obj)) {
                nextPathSegment = `[${key}]`
                yield* this.recursivelyChunk(obj[key], objIdentifier, currentPath + nextPathSegment)
            } else if (typeof obj[key] === 'object') {
                nextPathSegment = `.${key}`
                yield* this.recursivelyChunk(obj[key], objIdentifier, currentPath + nextPathSegment)
            } else {
                if (!this.pickKeysForEmbedding || (key in this.pickKeysForEmbedding)) {
                    let s = cleanString(JSON.stringify(obj[key]));
                    let entry: { path: string, preEmbedId?: string } = {
                        path: currentPath + `.${key}`
                    }

                    if ('id' in obj) {
                        entry.preEmbedId = obj['id']
                    }

                    yield {
                        pageContent: s,
                        metadata: {
                            type: <'JsonLoader'>'JsonLoader',
                            source: objIdentifier,
                            ...entry
                        }
                    }
                }
            }
        }

    }

    override async *getUnfilteredChunks() {
        const truncatedObjectString = truncateCenterString(JSON.stringify(this.object), 50);

        if (this.recurse) {
            yield* this.recursivelyChunk(this.object, truncatedObjectString, "$")
        } else {
            const array = Array.isArray(this.object) ? this.object : [this.object];

            let i = 0;
            for (const entry of array) {
                let s: string;
                if (this.pickKeysForEmbedding) {
                    const subset = Object.fromEntries(
                        this.pickKeysForEmbedding
                            .filter((key) => key in entry) // line can be removed to make it inclusive
                            .map((key) => [key, entry[key]]),
                    );
                    s = cleanString(JSON.stringify(subset));
                } else {
                    s = cleanString(JSON.stringify(entry));
                }

                if ('id' in entry) {
                    entry.preEmbedId = entry.id;
                    delete entry.id;
                }

                yield {
                    pageContent: s,
                    metadata: {
                        type: <'JsonLoader'>'JsonLoader',
                        source: truncatedObjectString,
                        ...entry,
                    },
                };

                i++;
            }
        }
    }
}
