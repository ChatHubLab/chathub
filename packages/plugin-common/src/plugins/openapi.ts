/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-namespace */
/* eslint-disable max-len */
import { StructuredTool, ToolParams } from '@langchain/core/tools'
import { Context } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import {
    fuzzyQuery,
    getMessageContent
} from 'koishi-plugin-chatluna/utils/string'
import { Config, logger } from '..'
import YAML from 'js-yaml'
import type { JSONSchema7 } from 'json-schema'
import { z, ZodObject, ZodSchema } from 'zod'

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    if (config.actions !== true) {
        return
    }

    const parsedActions = config.actionsList.flatMap((item) => {
        const spec = parseSpec(item.openAPISpec)

        return Object.entries(spec.paths).flatMap(([path, pathData]) =>
            Object.entries(pathData)
                .map(([method, operation]) => {
                    if (!isHttpMethod(method)) {
                        return null
                    }
                    return [
                        [
                            method,
                            operation as OpenAPIV3.OperationObject
                        ] as const,
                        [path, pathData as OpenAPIV3.PathItemObject] as const,
                        spec,
                        item
                    ] as const
                })
                .filter((item) => item !== null)
        )
    })

    for (const [operation, path, spec, action] of parsedActions) {
        const tool = OpenAPIPluginTool.fromAction({
            operation,
            path,
            spec,
            action,
            plugin
        })

        plugin.registerTool(tool.name, {
            selector(history) {
                return history.some((item) => {
                    const content = getMessageContent(item.content)

                    return fuzzyQuery(content, [
                        '令',
                        '调用',
                        '获取',
                        'get',
                        'help',
                        'command',
                        '执行',
                        '用',
                        'execute',
                        ...action.name.split('.'),
                        ...(action.selector ?? [])
                    ])
                })
            },

            async createTool(params, session) {
                return tool
            }
        })
    }
}

/**
 * Interface for parameters required to create an instance of
 * AIPluginTool.
 */
export interface AIPluginToolParams extends ToolParams {
    name: string
    description: string
    schema: ZodObject<any, any, any, any>
    meta: Meta
    plugin: ChatLunaPlugin
    parameters: FunctionParameter
    requestBody: JSONSchema7 | undefined
    action: Config['actionsList'][0]
}

// https://github.com/cloudflare/ai-utils/blob/main/src/createToolsFromOpenAPISpec.ts
export class OpenAPIPluginTool
    extends StructuredTool
    implements AIPluginToolParams
{
    static lc_name() {
        return 'OpenAPIluginTool'
    }

    name: string = ''
    description: string = ''
    schema: ZodObject<any, any, any, any>
    meta: Meta
    parameters: FunctionParameter
    requestBody: JSONSchema7 | undefined
    plugin: ChatLunaPlugin
    action: Config['actionsList'][0]

    constructor(params: AIPluginToolParams) {
        super(params)
        this.name = params.name
        this.description = params.description
        this.schema = params.schema
        this.meta = params.meta
        this.parameters = params.parameters
        this.requestBody = params.requestBody
        this.plugin = params.plugin
        this.action = params.action
    }

    /** @ignore */
    async _call(args: Record<string, any>) {
        const url = new URL(this.meta.url)
        const init: Parameters<this['plugin']['fetch']>[1] = {
            method: this.meta.method.toUpperCase(),
            headers: new Headers()
        }
        const queryParams = new URLSearchParams()
        const body: any = {}

        logger.debug('Initial args:', args)

        if (
            Object.keys(args).length > 0 &&
            !args.header &&
            !args.query &&
            !args.cookie &&
            !args.formData &&
            !args.body
        ) {
            // If args are there, but nothing else,
            // that means the AI might have hallucinated a query string inside the entire args object.
            args.query = args
        }

        // Apply config rules
        for (const key of Object.entries(this.action.headers)) {
            ;(init.headers as Headers).append(key[0], key[1])
        }

        logger.debug('URL before path replacement:', url.toString())

        // Decode URL to replace path parameters
        let decodedPathname = decodeURIComponent(url.pathname)

        for (const key in args.path) {
            if (decodedPathname.includes(`{${key}}`)) {
                decodedPathname = decodedPathname.replace(
                    `{${key}}`,
                    encodeURIComponent(args.path[key])
                )
            }
        }
        url.pathname = decodedPathname

        logger.debug('URL after path replacement:', url.toString())

        // Query parameters
        for (const key in args.query) {
            queryParams.append(key, args.query[key])
        }

        url.search = queryParams.toString()

        logger.debug('Query parameters:', url.search)

        // Headers
        for (const key in args.header) {
            ;(init.headers as Headers).append(key, args.header[key])
        }

        logger.debug('Headers:', init.headers)

        // Cookies
        if (args.cookie) {
            const cookieHeader = Object.entries(args.cookie)
                .map(([key, value]) => `${key}=${value}`)
                .join('; ')
            ;(init.headers as Headers).append('Cookie', cookieHeader)
        }

        // Body
        if (this.requestBody) {
            init.body = JSON.stringify(args.body)
            ;(init.headers as Headers).append(
                'Content-Type',
                'application/json'
            )
        } else if (Object.keys(body).length > 0) {
            init.body = JSON.stringify(body)
            ;(init.headers as Headers).append(
                'Content-Type',
                'application/json'
            )
        }

        logger.debug('Request body:', init.body)

        try {
            const res = await this.plugin.fetch(url.toString(), init)
            const result = await res.text()
            logger.debug('Response:', result)
            return result
        } catch (error) {
            if (error instanceof Error) {
                return JSON.stringify({ error: error.message })
            } else {
                return JSON.stringify({ error: String(error) })
            }
        }
    }

    static fromAction(
        action: Omit<
            AIPluginToolParams,
            | 'name'
            | 'description'
            | 'schema'
            | 'meta'
            | 'parameters'
            | 'requestBody'
        > & {
            path: readonly [string, OpenAPIV3.PathItemObject]
            operation: readonly [HttpMethod, OpenAPIV3.OperationObject]
            action: Config['actionsList'][number]
            spec: OpenAPIV3.Document
        }
    ) {
        const [path, pathData] = action.path
        const operation = action.operation[1]
        const openapiSpec = action.spec
        const url = getServerUrl(action.spec, pathData)
        const meta = {
            url: `${url.protocol}//${url.host}${url.pathname.replace(
                /\/$/,
                ''
            )}${path}`,
            method: action.operation[0]
        } satisfies Meta

        const parameters = extractParameters(pathData, operation, openapiSpec)
        const requestBody = extractRequestBody(operation, openapiSpec)

        const schemaShape: Record<string, z.ZodTypeAny> = {}

        if (Object.keys(parameters.path).length > 0) {
            const pathSchema = Object.entries(parameters.path).reduce(
                (acc, [key, param]) => {
                    acc[key] = param.required
                        ? z.string()
                        : z.string().optional()
                    return acc
                },
                {} as Record<string, ZodSchema>
            )
            schemaShape.path = z.object(pathSchema)
        }

        if (Object.keys(parameters.query).length > 0) {
            const querySchema = Object.entries(parameters.query).reduce(
                (acc, [key, param]) => {
                    acc[key] = param.required
                        ? z.string()
                        : z.string().optional()
                    return acc
                },
                {} as Record<string, ZodSchema>
            )
            schemaShape.query = z.object(querySchema)
        }

        if (Object.keys(parameters.header).length > 0) {
            const headerSchema = Object.entries(parameters.header).reduce(
                (acc, [key, param]) => {
                    acc[key] = param.required
                        ? z.string()
                        : z.string().optional()
                    return acc
                },
                {} as Record<string, ZodSchema>
            )
            schemaShape.header = z.object(headerSchema)
        }

        if (Object.keys(parameters.cookie).length > 0) {
            const cookieSchema = Object.entries(parameters.cookie).reduce(
                (acc, [key, param]) => {
                    acc[key] = param.required
                        ? z.string()
                        : z.string().optional()
                    return acc
                },
                {} as Record<string, ZodSchema>
            )
            schemaShape.cookie = z.object(cookieSchema)
        }

        if (Object.keys(parameters.formData).length > 0) {
            const formDataSchema = Object.entries(parameters.formData).reduce(
                (acc, [key, param]) => {
                    acc[key] = param.required
                        ? z.string()
                        : z.string().optional()
                    return acc
                },
                {} as Record<string, ZodSchema>
            )
            schemaShape.formData = z.object(formDataSchema)
        }

        if (Object.keys(parameters.body).length > 0) {
            const bodySchema = Object.entries(parameters.body).reduce(
                (acc, [key, param]) => {
                    acc[key] = param.required
                        ? z.string()
                        : z.string().optional()
                    return acc
                },
                {} as Record<string, ZodSchema>
            )
            schemaShape.body = z.object(bodySchema)
        }

        let normalizedName = generateRandomString()

        while (/^[0-9]/.test(normalizedName[0])) {
            normalizedName = generateRandomString()
        }

        return new OpenAPIPluginTool({
            name: normalizedName,
            description: action.action.description ?? operation.summary ?? '',
            schema: z.object(schemaShape),
            meta,
            parameters,
            requestBody,
            plugin: action.plugin,
            action: action.action
        })
    }
}

// From https://github.com/cloudflare/ai-utils/blob/main/src/createToolsFromOpenAPISpec.ts#L216

function generateRandomString(): string {
    return Math.random().toString(36).substring(7)
}

export type UppercaseHttpMethod =
    | 'GET'
    | 'POST'
    | 'PUT'
    | 'PATCH'
    | 'DELETE'
    | 'OPTIONS'
    | 'HEAD'
export type LowercaseHttpMethod =
    | 'get'
    | 'post'
    | 'put'
    | 'patch'
    | 'delete'
    | 'options'
    | 'head'

export type HttpMethod = UppercaseHttpMethod | LowercaseHttpMethod

export interface FunctionParameter {
    path: Record<string, JSONSchema7>
    query: Record<string, JSONSchema7>
    header: Record<string, JSONSchema7>
    cookie: Record<string, JSONSchema7>
    formData: Record<string, JSONSchema7>
    body: Record<string, JSONSchema7>
}

export interface Meta {
    url: string
    method: HttpMethod
}

function parseSpec(content: string): OpenAPIV3.Document {
    try {
        if (content.trim().startsWith('{')) {
            return JSON.parse(content) as OpenAPIV3.Document
        } else {
            return YAML.load(content) as OpenAPIV3.Document
        }
    } catch (error) {
        console.error('Error parsing the OpenAPI spec:', error)
        return undefined
    }
}

function getServerUrl(
    openapiSpec: OpenAPIV3.Document,
    pathData: OpenAPIV3.PathItemObject
): URL {
    let rawUrl = pathData.servers?.[0]?.url
    if (!rawUrl) {
        rawUrl = openapiSpec.servers?.[0]?.url
    }
    if (!rawUrl) {
        throw new Error('No server URL found in OpenAPI spec')
    }
    return new URL(rawUrl)
}

function isHttpMethod(method: string): method is HttpMethod {
    const httpMethods: HttpMethod[] = [
        'get',
        'post',
        'put',
        'patch',
        'delete',
        'options',
        'head'
    ]
    return httpMethods.includes(method.toLowerCase() as HttpMethod)
}

function extractParameters(
    pathData: OpenAPIV3.PathItemObject,
    operation: OpenAPIV3.OperationObject,
    openapiSpec: OpenAPIV3.Document
): FunctionParameter {
    const parameters: FunctionParameter = {
        path: {},
        query: {},
        header: {},
        cookie: {},
        formData: {},
        body: {}
    }

    const allParams = [
        ...(pathData.parameters || []),
        ...(operation.parameters || [])
    ]

    for (const param of allParams) {
        const resolvedParam = resolveReference(
            param,
            openapiSpec
        ) as OpenAPIV3.ParameterObject
        const paramInfo: JSONSchema7 = {
            type:
                (resolvedParam.schema as OpenAPIV3.SchemaObject).type ??
                'string',
            description: resolvedParam.description
        }

        parameters[resolvedParam.in as keyof FunctionParameter][
            resolvedParam.name
        ] = paramInfo as JSONSchema7
    }

    return parameters
}

function extractRequestBody(
    operation: OpenAPIV3.OperationObject,
    openapiSpec: OpenAPIV3.Document
): JSONSchema7 | undefined {
    if (!operation.requestBody) return undefined
    const resolvedBody = resolveReference(
        operation.requestBody,
        openapiSpec
    ) as OpenAPIV3.RequestBodyObject
    if (
        resolvedBody.content &&
        resolvedBody.content['application/json'] &&
        resolvedBody.content['application/json'].schema
    ) {
        return resolvedBody.content['application/json'].schema as JSONSchema7
    }
    return undefined
}

function resolveReference(
    ref: OpenAPIV3.ReferenceObject | any,
    openapiSpec: OpenAPIV3.Document
): any {
    if (!ref.$ref) return ref
    const refPath = ref.$ref.replace(/^#\//, '').split('/')
    return refPath.reduce(
        (acc: any, part: string) => acc && acc[part],
        openapiSpec
    )
}

// From: https://github.com/cloudflare/ai-utils/blob/main/src/types/openapi-schema.ts

export namespace OpenAPIV3_1 {
    type Modify<T, R> = Omit<T, keyof R> & R

    type PathsWebhooksComponents<T extends object = object> = {
        paths: PathsObject<T>
        webhooks: Record<string, PathItemObject | ReferenceObject>
        components: ComponentsObject
    }

    export type Document<T extends object = object> = Modify<
        Omit<OpenAPIV3.Document<T>, 'paths' | 'components'>,
        {
            info: InfoObject
            jsonSchemaDialect?: string
            servers?: ServerObject[]
        } & (
            | (Pick<PathsWebhooksComponents<T>, 'paths'> &
                  Omit<Partial<PathsWebhooksComponents<T>>, 'paths'>)
            | (Pick<PathsWebhooksComponents<T>, 'webhooks'> &
                  Omit<Partial<PathsWebhooksComponents<T>>, 'webhooks'>)
            | (Pick<PathsWebhooksComponents<T>, 'components'> &
                  Omit<Partial<PathsWebhooksComponents<T>>, 'components'>)
        )
    >

    export type InfoObject = Modify<
        OpenAPIV3.InfoObject,
        {
            summary?: string
            license?: LicenseObject
        }
    >

    export type ContactObject = OpenAPIV3.ContactObject

    export type LicenseObject = Modify<
        OpenAPIV3.LicenseObject,
        {
            identifier?: string
        }
    >

    export type ServerObject = Modify<
        OpenAPIV3.ServerObject,
        {
            url: string
            description?: string
            variables?: Record<string, ServerVariableObject>
        }
    >

    export type ServerVariableObject = Modify<
        OpenAPIV3.ServerVariableObject,
        {
            enum?: [string, ...string[]]
        }
    >

    export type PathsObject<
        T extends object = object,
        P extends object = object
    > = Record<string, (PathItemObject<T> & P) | undefined>

    export type HttpMethods = OpenAPIV3.HttpMethods

    export type PathItemObject<T extends object = object> = Modify<
        OpenAPIV3.PathItemObject<T>,
        {
            servers?: ServerObject[]
            parameters?: (ReferenceObject | ParameterObject)[]
        }
    > & {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        [method in HttpMethods]?: OperationObject<T>
    }

    export type OperationObject<T extends object = object> = Modify<
        OpenAPIV3.OperationObject<T>,
        {
            parameters?: (ReferenceObject | ParameterObject)[]
            requestBody?: ReferenceObject | RequestBodyObject
            responses?: ResponsesObject
            callbacks?: Record<string, ReferenceObject | CallbackObject>
            servers?: ServerObject[]
        }
    > &
        T

    export type ExternalDocumentationObject =
        OpenAPIV3.ExternalDocumentationObject

    export type ParameterObject = OpenAPIV3.ParameterObject

    export type HeaderObject = OpenAPIV3.HeaderObject

    export type ParameterBaseObject = OpenAPIV3.ParameterBaseObject

    export type NonArraySchemaObjectType =
        | OpenAPIV3.NonArraySchemaObjectType
        | 'null'

    export type ArraySchemaObjectType = OpenAPIV3.ArraySchemaObjectType

    /**
     * There is no way to tell typescript to require items when type is either 'array' or array containing 'array' type
     * 'items' will be always visible as optional
     * Casting schema object to ArraySchemaObject or NonArraySchemaObject will work fine
     */
    export type SchemaObject =
        | ArraySchemaObject
        | NonArraySchemaObject
        | MixedSchemaObject
        | boolean

    export interface ArraySchemaObject extends BaseSchemaObject {
        type: ArraySchemaObjectType
        items: ReferenceObject | SchemaObject
    }

    export interface NonArraySchemaObject extends BaseSchemaObject {
        type?: NonArraySchemaObjectType
    }

    interface MixedSchemaObject extends BaseSchemaObject {
        type?: (ArraySchemaObjectType | NonArraySchemaObjectType)[]
        items?: ReferenceObject | SchemaObject
    }

    export type BaseSchemaObject = Modify<
        Omit<OpenAPIV3.BaseSchemaObject, 'nullable'>,
        {
            examples?: OpenAPIV3.BaseSchemaObject['example'][]
            exclusiveMinimum?: boolean | number
            exclusiveMaximum?: boolean | number
            contentMediaType?: string
            $schema?: string
            additionalProperties?: boolean | ReferenceObject | SchemaObject
            properties?: {
                [name: string]: ReferenceObject | SchemaObject
            }
            allOf?: (ReferenceObject | SchemaObject)[]
            oneOf?: (ReferenceObject | SchemaObject)[]
            anyOf?: (ReferenceObject | SchemaObject)[]
            not?: ReferenceObject | SchemaObject
            discriminator?: DiscriminatorObject
            externalDocs?: ExternalDocumentationObject
            xml?: XMLObject
            const?: any
        }
    >

    export type DiscriminatorObject = OpenAPIV3.DiscriminatorObject

    export type XMLObject = OpenAPIV3.XMLObject

    export type ReferenceObject = Modify<
        OpenAPIV3.ReferenceObject,
        {
            summary?: string
            description?: string
        }
    >

    export type ExampleObject = OpenAPIV3.ExampleObject

    export type MediaTypeObject = Modify<
        OpenAPIV3.MediaTypeObject,
        {
            schema?: SchemaObject | ReferenceObject
            examples?: Record<string, ReferenceObject | ExampleObject>
        }
    >

    export type EncodingObject = OpenAPIV3.EncodingObject

    export type RequestBodyObject = Modify<
        OpenAPIV3.RequestBodyObject,
        {
            content: { [media: string]: MediaTypeObject }
        }
    >

    export type ResponsesObject = Record<
        string,
        ReferenceObject | ResponseObject
    >

    export type ResponseObject = Modify<
        OpenAPIV3.ResponseObject,
        {
            headers?: { [header: string]: ReferenceObject | HeaderObject }
            content?: { [media: string]: MediaTypeObject }
            links?: { [link: string]: ReferenceObject | LinkObject }
        }
    >

    export type LinkObject = Modify<
        OpenAPIV3.LinkObject,
        {
            server?: ServerObject
        }
    >

    export type CallbackObject = Record<
        string,
        PathItemObject | ReferenceObject
    >

    export type SecurityRequirementObject = OpenAPIV3.SecurityRequirementObject

    export type ComponentsObject = Modify<
        OpenAPIV3.ComponentsObject,
        {
            schemas?: Record<string, SchemaObject>
            responses?: Record<string, ReferenceObject | ResponseObject>
            parameters?: Record<string, ReferenceObject | ParameterObject>
            examples?: Record<string, ReferenceObject | ExampleObject>
            requestBodies?: Record<string, ReferenceObject | RequestBodyObject>
            headers?: Record<string, ReferenceObject | HeaderObject>
            securitySchemes?: Record<
                string,
                ReferenceObject | SecuritySchemeObject
            >
            links?: Record<string, ReferenceObject | LinkObject>
            callbacks?: Record<string, ReferenceObject | CallbackObject>
            pathItems?: Record<string, ReferenceObject | PathItemObject>
        }
    >

    export type SecuritySchemeObject = OpenAPIV3.SecuritySchemeObject

    export type HttpSecurityScheme = OpenAPIV3.HttpSecurityScheme

    export type ApiKeySecurityScheme = OpenAPIV3.ApiKeySecurityScheme

    export type OAuth2SecurityScheme = OpenAPIV3.OAuth2SecurityScheme

    export type OpenIdSecurityScheme = OpenAPIV3.OpenIdSecurityScheme

    export type TagObject = OpenAPIV3.TagObject
}

export namespace OpenAPIV3 {
    export interface Document<T extends object = object> {
        openapi: string
        info: InfoObject
        servers?: ServerObject[]
        paths: PathsObject<T>
        components?: ComponentsObject
        security?: SecurityRequirementObject[]
        tags?: TagObject[]
        externalDocs?: ExternalDocumentationObject
        'x-express-openapi-additional-middleware'?: (
            | ((request: any, response: any, next: any) => Promise<void>)
            | ((request: any, response: any, next: any) => void)
        )[]
        'x-express-openapi-validation-strict'?: boolean
    }

    export interface InfoObject {
        title: string
        description?: string
        termsOfService?: string
        contact?: ContactObject
        license?: LicenseObject
        version: string
    }

    export interface ContactObject {
        name?: string
        url?: string
        email?: string
    }

    export interface LicenseObject {
        name: string
        url?: string
    }

    export interface ServerObject {
        url: string
        description?: string
        variables?: { [variable: string]: ServerVariableObject }
    }

    export interface ServerVariableObject {
        enum?: string[] | number[]
        default: string | number
        description?: string
    }

    export interface PathsObject<
        T extends object = object,
        P extends object = object
    > {
        [pattern: string]: (PathItemObject<T> & P) | undefined
    }

    // All HTTP methods allowed by OpenAPI 3 spec
    // See https://swagger.io/specification/#path-item-object
    // You can use keys or values from it in TypeScript code like this:
    //     for (const method of Object.values(OpenAPIV3.HttpMethods)) { … }
    export enum HttpMethods {
        GET = 'get',
        PUT = 'put',
        POST = 'post',
        DELETE = 'delete',
        OPTIONS = 'options',
        HEAD = 'head',
        PATCH = 'patch',
        TRACE = 'trace'
    }

    export type PathItemObject<T extends object = object> = {
        $ref?: string
        summary?: string
        description?: string
        servers?: ServerObject[]
        parameters?: (ReferenceObject | ParameterObject)[]
    } & {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        [method in HttpMethods]?: OperationObject<T>
    }

    export type OperationObject<T extends object = object> = {
        tags?: string[]
        summary?: string
        description?: string
        externalDocs?: ExternalDocumentationObject
        operationId?: string
        parameters?: (ReferenceObject | ParameterObject)[]
        requestBody?: ReferenceObject | RequestBodyObject
        responses: ResponsesObject
        callbacks?: { [callback: string]: ReferenceObject | CallbackObject }
        deprecated?: boolean
        security?: SecurityRequirementObject[]
        servers?: ServerObject[]
    } & T

    export interface ExternalDocumentationObject {
        description?: string
        url: string
    }

    export interface ParameterObject extends ParameterBaseObject {
        name: string
        in: string
    }

    export interface HeaderObject extends ParameterBaseObject {}

    export interface ParameterBaseObject {
        description?: string
        required?: boolean
        deprecated?: boolean
        allowEmptyValue?: boolean
        style?: string
        explode?: boolean
        allowReserved?: boolean
        schema?: ReferenceObject | SchemaObject
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        example?: any
        examples?: { [media: string]: ReferenceObject | ExampleObject }
        content?: { [media: string]: MediaTypeObject }
    }
    export type NonArraySchemaObjectType =
        | 'boolean'
        | 'object'
        | 'number'
        | 'string'
        | 'integer'
    export type ArraySchemaObjectType = 'array'
    export type SchemaObject = ArraySchemaObject | NonArraySchemaObject

    export interface ArraySchemaObject extends BaseSchemaObject {
        type: ArraySchemaObjectType
        items: ReferenceObject | SchemaObject
    }

    export interface NonArraySchemaObject extends BaseSchemaObject {
        type?: NonArraySchemaObjectType
    }

    export interface BaseSchemaObject {
        // JSON schema allowed properties, adjusted for OpenAPI
        title?: string
        description?: string
        format?: string
        default?: any
        multipleOf?: number
        maximum?: number
        exclusiveMaximum?: boolean
        minimum?: number
        exclusiveMinimum?: boolean
        maxLength?: number
        minLength?: number
        pattern?: string
        additionalProperties?: boolean | ReferenceObject | SchemaObject
        maxItems?: number
        minItems?: number
        uniqueItems?: boolean
        maxProperties?: number
        minProperties?: number
        required?: string[]
        enum?: any[]
        properties?: {
            [name: string]: ReferenceObject | SchemaObject
        }
        allOf?: (ReferenceObject | SchemaObject)[]
        oneOf?: (ReferenceObject | SchemaObject)[]
        anyOf?: (ReferenceObject | SchemaObject)[]
        not?: ReferenceObject | SchemaObject

        // OpenAPI-specific properties
        nullable?: boolean
        discriminator?: DiscriminatorObject
        readOnly?: boolean
        writeOnly?: boolean
        xml?: XMLObject
        externalDocs?: ExternalDocumentationObject
        example?: any
        deprecated?: boolean
    }

    export interface DiscriminatorObject {
        propertyName: string
        mapping?: { [value: string]: string }
    }

    export interface XMLObject {
        name?: string
        namespace?: string
        prefix?: string
        attribute?: boolean
        wrapped?: boolean
    }

    export interface ReferenceObject {
        $ref: string
    }

    export interface ExampleObject {
        summary?: string
        description?: string
        value?: any
        externalValue?: string
    }

    export interface MediaTypeObject {
        schema?: ReferenceObject | SchemaObject
        example?: any
        examples?: { [media: string]: ReferenceObject | ExampleObject }
        encoding?: { [media: string]: EncodingObject }
    }

    export interface EncodingObject {
        contentType?: string
        headers?: { [header: string]: ReferenceObject | HeaderObject }
        style?: string
        explode?: boolean
        allowReserved?: boolean
    }

    export interface RequestBodyObject {
        description?: string
        content: { [media: string]: MediaTypeObject }
        required?: boolean
    }

    export interface ResponsesObject {
        [code: string]: ReferenceObject | ResponseObject
    }

    export interface ResponseObject {
        description: string
        headers?: { [header: string]: ReferenceObject | HeaderObject }
        content?: { [media: string]: MediaTypeObject }
        links?: { [link: string]: ReferenceObject | LinkObject }
    }

    export interface LinkObject {
        operationRef?: string
        operationId?: string
        parameters?: { [parameter: string]: any }
        requestBody?: any
        description?: string
        server?: ServerObject
    }

    export interface CallbackObject {
        [url: string]: PathItemObject
    }

    export interface SecurityRequirementObject {
        [name: string]: string[]
    }

    export interface ComponentsObject {
        schemas?: { [key: string]: ReferenceObject | SchemaObject }
        responses?: { [key: string]: ReferenceObject | ResponseObject }
        parameters?: { [key: string]: ReferenceObject | ParameterObject }
        examples?: { [key: string]: ReferenceObject | ExampleObject }
        requestBodies?: { [key: string]: ReferenceObject | RequestBodyObject }
        headers?: { [key: string]: ReferenceObject | HeaderObject }
        securitySchemes?: {
            [key: string]: ReferenceObject | SecuritySchemeObject
        }
        links?: { [key: string]: ReferenceObject | LinkObject }
        callbacks?: { [key: string]: ReferenceObject | CallbackObject }
    }

    export type SecuritySchemeObject =
        | HttpSecurityScheme
        | ApiKeySecurityScheme
        | OAuth2SecurityScheme
        | OpenIdSecurityScheme

    export interface HttpSecurityScheme {
        type: 'http'
        description?: string
        scheme: string
        bearerFormat?: string
    }

    export interface ApiKeySecurityScheme {
        type: 'apiKey'
        description?: string
        name: string
        in: string
    }

    export interface OAuth2SecurityScheme {
        type: 'oauth2'
        description?: string
        flows: {
            implicit?: {
                authorizationUrl: string
                refreshUrl?: string
                scopes: { [scope: string]: string }
            }
            password?: {
                tokenUrl: string
                refreshUrl?: string
                scopes: { [scope: string]: string }
            }
            clientCredentials?: {
                tokenUrl: string
                refreshUrl?: string
                scopes: { [scope: string]: string }
            }
            authorizationCode?: {
                authorizationUrl: string
                tokenUrl: string
                refreshUrl?: string
                scopes: { [scope: string]: string }
            }
        }
    }

    export interface OpenIdSecurityScheme {
        type: 'openIdConnect'
        description?: string
        openIdConnectUrl: string
    }

    export interface TagObject {
        name: string
        description?: string
        externalDocs?: ExternalDocumentationObject
    }
}

export interface IJsonSchema {
    id?: string
    $schema?: string
    title?: string
    description?: string
    multipleOf?: number
    maximum?: number
    exclusiveMaximum?: boolean
    minimum?: number
    exclusiveMinimum?: boolean
    maxLength?: number
    minLength?: number
    pattern?: string
    additionalItems?: boolean | IJsonSchema
    items?: IJsonSchema | IJsonSchema[]
    maxItems?: number
    minItems?: number
    uniqueItems?: boolean
    maxProperties?: number
    minProperties?: number
    required?: string[]
    additionalProperties?: boolean | IJsonSchema
    definitions?: {
        [name: string]: IJsonSchema
    }
    properties?: {
        [name: string]: IJsonSchema
    }
    patternProperties?: {
        [name: string]: IJsonSchema
    }
    dependencies?: {
        [name: string]: IJsonSchema | string[]
    }
    enum?: any[]
    type?: string | string[]
    allOf?: IJsonSchema[]
    anyOf?: IJsonSchema[]
    oneOf?: IJsonSchema[]
    not?: IJsonSchema
    $ref?: string
}
