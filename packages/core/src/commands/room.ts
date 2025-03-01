import { Context } from 'koishi'
import { Config } from '../config'
import { ChatChain } from '../chains/chain'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    ctx.command('chatluna.room')

    ctx.command('chatluna.room.create')
        .option('name', '-n <name:string>')
        .option('preset', '-p <preset:string>')
        .option('model', '-m <model:string>')
        .option('chatMode', '-c <chatMode:string>')
        .option('password', '-w <password:string>')
        .option('visibility', '-v <visibility:string>')
        .action(async ({ session, options }) => {
            await chain.receiveCommand(session, 'create_room', {
                room_resolve: {
                    name: options.name ?? undefined,
                    preset: options.preset ?? undefined,
                    model: options.model ?? undefined,
                    chatMode: options.chatMode ?? undefined,
                    password: options.password ?? undefined,
                    visibility: options.visibility ?? undefined
                }
            })
        })

    ctx.command('chatluna.room.delete [room:text]').action(
        async ({ session }, room) => {
            await chain.receiveCommand(session, 'delete_room', {
                room_resolve: {
                    name: room
                }
            })
        }
    )

    ctx.command('chatluna.room.auto-update <status:string>')
        .option('room', '-r <room:string>')
        .action(async ({ session, options }, status) => {
            if (
                status.toLowerCase() !== 'true' &&
                status.toLowerCase() !== 'false'
            ) {
                return session.text('.messages.invalid-status')
            }

            await chain.receiveCommand(session, 'set_auto_update_room', {
                room_resolve: {
                    name: options.room
                },
                auto_update_room: status.toLowerCase() === 'true'
            })
        })

    ctx.command('chatluna.room.kick <...arg:user>').action(
        async ({ session }, ...user) => {
            const users = user.map((u) => u.split(':')[1])
            await chain.receiveCommand(session, 'kick_member', {
                resolve_user: {
                    id: users
                }
            })
        }
    )

    ctx.command('chatluna.room.invite <...arg:user>').action(
        async ({ session }, ...user) => {
            const users = user.map((u) => u.split(':')[1])
            await chain.receiveCommand(session, 'invite_room', {
                resolve_user: {
                    id: users
                }
            })
        }
    )

    ctx.command('chatluna.room.join <id:text>').action(
        async ({ session }, name) => {
            await chain.receiveCommand(session, 'join_room', {
                room_resolve: {
                    name
                }
            })
        }
    )

    ctx.command('chatluna.room.leave [room:text]').action(
        async ({ session, options }, room) => {
            await chain.receiveCommand(session, 'leave_room', {
                room_resolve: {
                    name: room,
                    id: room
                }
            })
        }
    )

    ctx.command('chatluna.room.clear [room:text]')
        .alias('chatluna.chat.clear')
        .action(async ({ session }, room) => {
            await chain.receiveCommand(session, 'clear_room', {
                room_resolve: {
                    name: room
                }
            })
        })

    ctx.command('chatluna.room.set')
        .option('name', '-n <name:string>')
        .option('preset', '-p <preset:string>')
        .option('model', '-m <model:string>')
        .option('chatMode', '-c <chatMode:string>')
        .option('password', '-w <password:string>')
        .option('visibility', '-v <visibility:string>')
        .action(async ({ session, options }) => {
            await chain.receiveCommand(session, 'set_room', {
                room_resolve: {
                    name: options.name ?? undefined,
                    preset: options.preset ?? undefined,
                    model: options.model ?? undefined,
                    chatMode: options.chatMode ?? undefined,
                    password: options.password ?? undefined,
                    visibility: options.visibility ?? undefined
                }
            })
        })

    ctx.command('chatluna.room.list')
        .option('page', '-p <page:number>')
        .option('limit', '-l <limit:number>')
        .option('all', '-a')
        .action(async ({ options, session }) => {
            await chain.receiveCommand(session, 'list_room', {
                page: options.page ?? 1,
                limit: options.limit ?? 2,
                all_room: options.all ?? false
            })
        })

    ctx.command('chatluna.room.transfer <arg:user>').action(
        async ({ session }, user) => {
            await chain.receiveCommand(session, 'transfer_room', {
                resolve_user: {
                    id: user.split(':')[1]
                }
            })
        }
    )

    ctx.command('chatluna.room.info [room:text]').action(
        async ({ session }, room) => {
            await chain.receiveCommand(session, 'room_info', {
                room_resolve: {
                    name: room
                }
            })
        }
    )

    ctx.command('chatluna.room.switch <name:text>').action(
        async ({ session }, name) => {
            await chain.receiveCommand(session, 'switch_room', {
                room_resolve: {
                    name,
                    id: name
                }
            })
        }
    )

    ctx.command('chatluna.room.permission <user:user>').action(
        async ({ session }, user) => {
            await chain.receiveCommand(session, 'room_permission', {
                resolve_user: {
                    id: user.split(':')[1]
                }
            })
        }
    )

    ctx.command('chatluna.room.mute <...user:user>')
        .option('room', '-r <room:string>')
        .action(async ({ session, options }, ...user) => {
            await chain.receiveCommand(session, 'mute_user', {
                room_resolve: {
                    name: options.room
                },
                resolve_user: {
                    id: user.map((u) => u.split(':')[1])
                }
            })
        })
}
