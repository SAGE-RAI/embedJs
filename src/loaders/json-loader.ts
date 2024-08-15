/**
 * author: joseph.kwarteng@open.ac.uk
 * created on: 15-08-2024-12h-17m
 * github: https://github.com/kwartengj
 * copyright 2024
*/

import md5 from 'md5';

import { BaseLoader } from '../interfaces/base-loader.js';
import { cleanString, truncateCenterString } from '../util/strings.js';

export class JsonLoader extends BaseLoader<{ type: 'JsonLoader' }> {
    private readonly object: Record<string, unknown> | Record<string, unknown>[];
    private readonly pickKeysForEmbedding?: string[];

    constructor({
        object,
        pickKeysForEmbedding,
    }: {
        object: Record<string, unknown> | Record<string, unknown>[];
        pickKeysForEmbedding?: string[];
    }) {
        super(`JsonLoader_${md5(cleanString(JSON.stringify(object)))}`, {
            object: truncateCenterString(JSON.stringify(object), 50),
        });

        this.pickKeysForEmbedding = pickKeysForEmbedding;
        this.object = object;
    }

    // Utility function to recursively extract all string values along with their paths
    private extractStrings(
        obj: Record<string, unknown>,
        parentPath = ''
    ): { path: string; value: string }[] {
        const results: { path: string; value: string }[] = [];

        const recursiveSearch = (item: unknown, currentPath: string) => {
            if (typeof item === 'string') {
                results.push({ path: currentPath, value: item });
            } else if (typeof item === 'object' && item !== null) {
                for (const key in item) {
                    if (Object.prototype.hasOwnProperty.call(item, key)) {
                        recursiveSearch((item as Record<string, unknown>)[key], `${currentPath}/${key}`);
                    }
                }
            }
        };

        recursiveSearch(obj, parentPath);
        return results;
    }

    override async *getUnfilteredChunks() {
        const truncatedObjectString = truncateCenterString(JSON.stringify(this.object), 50);
        const array = Array.isArray(this.object) ? this.object : [this.object];

        let i = 0;
        for (const entry of array) {
            let stringsWithPaths: { path: string; value: string }[];

            if (this.pickKeysForEmbedding) {
                const subset = Object.fromEntries(
                    this.pickKeysForEmbedding
                        .filter((key) => key in entry)
                        .map((key) => [key, entry[key]])
                );
                stringsWithPaths = this.extractStrings(subset);
            } else {
                stringsWithPaths = this.extractStrings(entry);
            }

            for (const { path, value } of stringsWithPaths) {
                const cleanedValue = cleanString(value);

                yield {
                    pageContent: cleanedValue,
                    metadata: {
                        type: <'JsonLoader'>'JsonLoader',
                        source: truncatedObjectString,
                        path: path || '/', // The root path if path is empty
                        index: i,
                    },
                };
            }

            i++;
        }
    }
}

