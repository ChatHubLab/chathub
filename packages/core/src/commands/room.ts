import { Context, Session } from 'koishi';
import { Config } from '../config';
import { ChatChain } from '../chains/chain';

export function apply(ctx: Context, config: Config, chain: ChatChain) {


    ctx.command('chathub.room', 'chathub 房间相关指令', {
        authority: 1,
    })

    ctx.command("chathub.room.create", "创建一个新房间")
        .alias("创建房间")
        .option("name", "-n <name:string> 房间名字")
        .option("preset", "-p <preset:string> 房间预设")
        .option("model", "-m <model:string> 房间模型")
        .option("chatMode", "-c <chatMode:string> 房间聊天模式")
        .option("password", "-w <password:string> 房间密码")
        .option("visibility", "-v <visibility:string> 房间可见性")
        .action(async ({ session, options }) => {
            await chain.receiveCommand(
                session, "createRoom", {
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

    ctx.command("chathub.room.delete [room:text]", "删除一个房间")
        .alias("删除房间")
        .action(async ({ session }, room) => {
            await chain.receiveCommand(
                session, "deleteRoom",
                {
                    room_resolve: {
                        name: room,
                    }
                }
            )
        })

    ctx.command("chathub.room.kick <...arg:user>", "踢出某个人员在你当前的房间")
        .alias("踢出成员")
        .action(async ({ session }, ...user) => {
            const users = user.map(u => u.split(":")[1])
            await chain.receiveCommand(
                session, "kickMember", {
                resolve_user: {
                    id: users
                }
            }
            )
        })

    ctx.command("chathub.room.invite <...arg:user>", "邀请进入房间")
        .alias("邀请进房")
        .action(async ({ session }, ...user) => {
            const users = user.map(u => u.split(":")[1])
            await chain.receiveCommand(
                session, "inviteRoom", {
                resolve_user: {
                    id: users
                }
            })
        })

    ctx.command("chathub.room.join <id:text>", "加入某个房间")
        .alias("加入房间")
        .action(async ({ session }, name) => {
            await chain.receiveCommand(
                session, "joinRoom", {
                room_resolve: {
                    name
                }
            })
        })

    ctx.command("chathub.room.add_to_group <id:string>", "允许房间在某个群里也可以使用")
        .option("group", "-g <group:string> 群号")
        .alias("加入房间到群")
        .action(async ({ session, options }, name) => {
            await chain.receiveCommand(
                session, "addRoomToGroup", {
                room_resolve: {
                    name: options.group
                },
                resolve_user: {
                    id: name
                }
            })
        })


    ctx.command("chathub.room.leave [room:text]", "离开当前房间")
        .alias("离开房间")
        .action(async ({ session, options }, room) => {
            await chain.receiveCommand(
                session, "leaveRoom", {
                room_resolve: {
                    name: room,
                    id: room
                }
            })
        })

    ctx.command("chathub.room.clear [room:text]", "清除房间的聊天记录")
        .alias("清除记录")
        .action(async ({ session }, room) => {
            await chain.receiveCommand(
                session, "clearRoom", {
                room_resolve: {
                    name: room,
                }
            })
        })


    ctx.command("chathub.room.set", "设置房间的属性")
        .alias("设置房间")
        .option("name", "-n <name:string> 房间名字")
        .option("preset", "-p <preset:string> 房间预设")
        .option("model", "-m <model:string> 房间模型")
        .option("chatMode", "-c <chatMode:string> 房间聊天模式")
        .option("password", "-w <password:string> 房间密码")
        .option("visibility", "-v <visibility:string> 房间可见性")
        .action(async ({ session, options }) => {
            await chain.receiveCommand(
                session, "setRoom", {
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

    ctx.command("chathub.room.list", "列出所有你加入的房间")
        .alias("房间列表")
        .action(async ({ session }) => {
            await chain.receiveCommand(
                session, "listRoom"
            )
        })

    ctx.command("chathub.room.transfer <arg:user>", "转移房间的房主")
        .alias("转移房主")
        .action(async ({ session }, user) => {
            await chain.receiveCommand(
                session, "transferRoom", {
                resolve_user: {
                    id: user.split(":")[1]
                }
            })
        })

    ctx.command("chathub.room.info [room:text]", "查看当前房间的信息")
        .alias("房间信息")
        .action(async ({ session }, room) => {
            await chain.receiveCommand(
                session, "roomInfo", {
                room_resolve: {
                    name: room,
                }
            })
        })

    ctx.command("chathub.room.switch <name:text>", "切换到你已经加入了的房间")
        .alias("切换房间")
        .action(async ({ session }, name) => {
            await chain.receiveCommand(
                session, "switchRoom",
                {
                    room_resolve: {
                        name: name,
                        id: name
                    }
                }
            )
        })

    ctx.command("chathub.room.permission <user:user>", "修改房间里某人的权限")
        .alias("修改房间用户权限")
        .action(async ({ session }, user) => {
            await chain.receiveCommand(
                session, "roomPermission", {
                resolve_user: {
                    id: user.split(":")[1]
                }
            })
        })

    ctx.command("chathub.room.mute <...user:user>", "禁言某个用户，不让其发言")
        .option("room", "-r <room:string> 指定房间")
        .alias("禁言房间用户")
        .action(async ({ session, options }, ...user) => {
            await chain.receiveCommand(
                session, "muteUser", {
                room_resolve: {
                    name: options.room
                },
                resolve_user: {
                    id: user.map(u => u.split(":")[1])
                }
            })
        })
}