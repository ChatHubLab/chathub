// Types representing the OpenAI function definitions. While the OpenAI client library
// does have types for function definitions, the properties are just Record<string, unknown>,

import { StructuredTool } from '@langchain/core/tools'
import { zodToJsonSchema } from 'zod-to-json-schema'

interface ObjectProp {
    type: 'object'
    properties?: {
        [key: string]: Prop
    }
    required?: string[]
}

interface AnyOfProp {
    anyOf: Prop[]
}

type Prop = {
    description?: string
} & (
    | AnyOfProp
    | ObjectProp
    | {
          type: 'string'
          enum?: string[]
      }
    | {
          type: 'number' | 'integer'
          minimum?: number
          maximum?: number
          enum?: number[]
      }
    | { type: 'boolean' }
    | { type: 'null' }
    | {
          type: 'array'
          items?: Prop
      }
)

function isAnyOfProp(prop: Prop): prop is AnyOfProp {
    return (
        (prop as AnyOfProp).anyOf !== undefined &&
        Array.isArray((prop as AnyOfProp).anyOf)
    )
}

// When OpenAI use functions in the prompt, they format them as TypeScript definitions rather than OpenAPI JSON schemas.
// This function converts the JSON schemas into TypeScript definitions.
export function formatFunctionDefinitions(functions: StructuredTool[]) {
    const lines = ['namespace functions {', '']
    for (const f of functions) {
        if (f.description) {
            lines.push(`// ${f.description}`)
        }

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const schema = zodToJsonSchema(f.schema as any)
        if (Object.keys(schema ?? {}).length > 0) {
            lines.push(`type ${f.name} = (_: {`)
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            lines.push(formatObjectProperties(schema as any, 0))
            lines.push('}) => any;')
        } else {
            lines.push(`type ${f.name} = () => any;`)
        }
        lines.push('')
    }
    lines.push('} // namespace functions')
    return lines.join('\n')
}

// Format just the properties of an object (not including the surrounding braces)
function formatObjectProperties(obj: ObjectProp, indent: number): string {
    const lines: string[] = []
    for (const [name, param] of Object.entries(obj.properties ?? {})) {
        if (param.description && indent < 2) {
            lines.push(`// ${param.description}`)
        }
        if (obj.required?.includes(name)) {
            lines.push(`${name}: ${formatType(param, indent)},`)
        } else {
            lines.push(`${name}?: ${formatType(param, indent)},`)
        }
    }
    return lines.map((line) => ' '.repeat(indent) + line).join('\n')
}

// Format a single property type
function formatType(param: Prop, indent: number): string {
    if (isAnyOfProp(param)) {
        return param.anyOf.map((v) => formatType(v, indent)).join(' | ')
    }
    switch (param.type) {
        case 'string':
            if (param.enum) {
                return param.enum.map((v) => `"${v}"`).join(' | ')
            }
            return 'string'
        case 'number':
            if (param.enum) {
                return param.enum.map((v) => `${v}`).join(' | ')
            }
            return 'number'
        case 'integer':
            if (param.enum) {
                return param.enum.map((v) => `${v}`).join(' | ')
            }
            return 'number'
        case 'boolean':
            return 'boolean'
        case 'null':
            return 'null'
        case 'object':
            return ['{', formatObjectProperties(param, indent + 2), '}'].join(
                '\n'
            )
        case 'array':
            if (param.items) {
                return `${formatType(param.items, indent)}[]`
            }
            return 'any[]'
        default:
            return ''
    }
}
