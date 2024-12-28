import { Context } from 'koishi'
import { Config } from '../config'
import { ChatChain } from '../chains/chain'
import { RenderType } from '../types'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    ctx.command('chatluna', {
        authority: 1
    }).alias('chatluna')

    ctx.command('chatluna.chat', {
        authority: 1
    })

    ctx.command('chatluna.chat.text <message:text>')
        .option('room', '-r <room:string>')
        .option('type', '-t <type: string>')
        .action(async ({ options, session }, message) => {
            const renderType = options.type ?? config.outputMode

            if (
                ![
                    'raw',
                    'voice',
                    'text',
                    'image',
                    'mixed-image',
                    'mixed-voice'
                ].some((type) => type === renderType)
            ) {
                return session.text('.invalid-render-type')
            }

            await chain.receiveCommand(session, '', {
                message,
                room_resolve: {
                    name: options.room
                },
                renderOptions: {
                    split: config.splitMessage,
                    type: renderType as RenderType
                }
            })
        })

    ctx.command('chatluna.chat.rollback [message:text]')
        .option('room', '-r <room:string>')
        .action(async ({ options, session }, message) => {
            await chain.receiveCommand(session, 'rollback', {
                message,
                room_resolve: {
                    name: options.room
                },
                renderOptions: {
                    split: config.splitMessage,
                    type: config.outputMode as RenderType
                }
            })
        })

    ctx.command('chatluna.chat.stop')
        .option('room', '-r <room:string>')
        .action(async ({ options, session }, message) => {
            await chain.receiveCommand(session, 'stop_chat', {
                room_resolve: {
                    name: options.room
                }
            })
        })

    ctx.command('chatluna.chat.voice <message:text>')
        .option('room', '-r <room:string>')
        .option('speaker', '-s <speakerId:number>', { authority: 1 })
        .action(async ({ options, session }, message) => {
            await chain.receiveCommand(session, '', {
                message,
                renderOptions: {
                    split: config.splitMessage,
                    type: 'voice',
                    voice: {
                        speakerId: options.speaker
                    }
                },
                room_resolve: {
                    name: options.room
                }
            })
        })

    ctx.command('chatluna.wipe', { authority: 3 }).action(
        async ({ session }) => {
            await chain.receiveCommand(session, 'wipe')
        }
    )

    ctx.command('chatluna.restart').action(async ({ options, session }) => {
        await chain.receiveCommand(session, 'restart')
    })
}
