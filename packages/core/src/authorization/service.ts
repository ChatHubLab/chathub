import { Context, Service, Session } from 'koishi'
import { Config } from '../config'
import { ChatHubAuthGroup, ChatHubAuthUser } from './types'
import { ChatHubError, ChatHubErrorCode } from '../utils/error'
import { createLogger } from '../utils/logger'

const logger = createLogger()

export class ChatHubAuthService extends Service {
    constructor(
        public readonly ctx: Context,
        public config: Config
    ) {
        super(ctx, 'chathub_auth')

        ctx.on('ready', async () => {
            await this._defineDatabase()
        })
    }

    async getAccount(
        session: Session,
        userId: string = session.userId
    ): Promise<ChatHubAuthUser> {
        const list = await this.ctx.database.get('chathub_auth_user', {
            userId
        })

        if (list.length === 0) {
            return this._createAccount(session, userId)
        } else if (list.length > 1) {
            throw new ChatHubError(ChatHubErrorCode.USER_NOT_FOUND)
        }

        return list[0]
    }

    private async _createAccount(
        session: Session,
        userId: string = session.userId
    ): Promise<ChatHubAuthUser> {
        const user = await this.ctx.database.getUser(session.platform, userId)

        if (user == null) {
            throw new ChatHubError(
                ChatHubErrorCode.USER_NOT_FOUND,
                new Error(`
                user not found in platform ${session.platform} and id ${userId}`)
            )
        }

        // TODO: automatic grant of user type
        const authType =
            user.authority > 2 ? 'admin' : user.authority > 1 ? 'user' : 'guest'

        const authUser: ChatHubAuthUser = {
            userId,
            balance:
                authType === 'admin' ? 10000 : authType === 'user' ? 100 : 1,
            authType
        }

        await this.ctx.database.upsert('chathub_auth_user', [authUser])

        await this.addUserToGroup(authUser, authType)

        return authUser
    }

    async getAuthGroup(
        session: Session,
        platform: string,
        userId: string = session.userId
    ): Promise<ChatHubAuthGroup> {
        // search platform

        const groups = (
            await this.ctx.database.get('chathub_auth_group', {
                platform: {
                    $or: [undefined, platform]
                }
            })
        ).sort((a, b) => {
            // prefer the same platform
            if (a.platform === platform) {
                return -1
            }
            if (b.platform === platform) {
                return 1
            }
            return a.priority - b.priority
        })

        // Here there will be no such thing as a user joining too many groups, so a query will work.
        const groupIds = groups.map((g) => g.id)

        const joinedGroups = (
            await this.ctx.database.get('chathub_auth_joined_user', {
                groupId: {
                    // max 50??
                    $in: groupIds
                },
                userId
            })
        ).sort(
            (a, b) => groupIds.indexOf(a.groupId) - groupIds.indexOf(b.groupId)
        )

        if (joinedGroups.length === 0) {
            throw new ChatHubError(ChatHubErrorCode.AUTH_GROUP_NOT_JOINED)
        }

        logger.debug(groups)
        logger.debug(joinedGroups)

        const result = groups.find((g) => g.id === joinedGroups[0].groupId)

        if (result == null) {
            throw new ChatHubError(
                ChatHubErrorCode.AUTH_GROUP_NOT_FOUND,
                new Error(`Group not found for user ${session.username} and platform
                ${platform}`)
            )
        }

        return result
    }

    async getAuthGroups(platform?: string) {
        const groups = await this.ctx.database.get('chathub_auth_group', {
            platform
        })

        return groups
    }

    async calculateBalance(
        session: Session,
        platform: string,
        usedTokenNumber: number,
        userId: string = session.userId
    ): Promise<number> {
        // TODO: use default balance checker
        await this.getAccount(session)

        const currentAuthGroup = await this.getAuthGroup(
            session,
            platform,
            userId
        )

        // 1k token per
        const usedBalance =
            currentAuthGroup.constPerToken * (usedTokenNumber / 1000)

        return await this.modifyBalance(session, -usedBalance)
    }

    async getBalance(
        session: Session,
        userId: string = session.userId
    ): Promise<number> {
        return (await this.getAccount(session, userId)).balance
    }

    async modifyBalance(
        session: Session,
        amount: number,
        userId: string = session.userId
    ): Promise<number> {
        const user = await this.getAccount(session, userId)

        user.balance += amount

        await this.ctx.database.upsert('chathub_auth_user', [user])

        return user.balance
    }

    private async _getAuthGroup(authGroupId: number) {
        const authGroup = (
            await this.ctx.database.get('chathub_auth_group', {
                id: authGroupId
            })
        )?.[0]

        if (authGroup == null) {
            throw new ChatHubError(
                ChatHubErrorCode.AUTH_GROUP_NOT_FOUND,
                new Error(`Auth group not found for id ${authGroupId}`)
            )
        }

        return authGroup
    }

    async resetAuthGroup(authGroupId: number) {
        const authGroup = await this._getAuthGroup(authGroupId)
        const currentTime = new Date()

        authGroup.lastCallTime = authGroup.lastCallTime ?? currentTime.getTime()

        const authGroupDate = new Date(authGroup.lastCallTime)

        const currentDayOfStart = new Date().setHours(0, 0, 0, 0)

        // If the last call time is not today, then all zeroed out

        if (authGroupDate.getTime() < currentDayOfStart) {
            authGroup.currentLimitPerDay = 0
            authGroup.currentLimitPerMin = 0
            authGroup.lastCallTime = currentTime.getTime()

            await this.ctx.database.upsert('chathub_auth_group', [authGroup])

            return authGroup
        }

        // Check to see if it's been more than a minute since the last call

        if (currentTime.getTime() - authGroup.lastCallTime >= 60000) {
            // clear

            authGroup.currentLimitPerMin = 0
            authGroup.lastCallTime = currentTime.getTime()

            await this.ctx.database.upsert('chathub_auth_group', [authGroup])

            return authGroup
        }

        return authGroup
    }

    async increaseAuthGroupCount(authGroupId: number) {
        const authGroup = await this._getAuthGroup(authGroupId)
        const currentTime = new Date()

        authGroup.lastCallTime = authGroup.lastCallTime ?? currentTime.getTime()

        const authGroupDate = new Date(authGroup.lastCallTime)

        const currentDayOfStart = new Date().setHours(0, 0, 0, 0)

        // If the last call time is not today, then all zeroed out

        if (authGroupDate.getTime() < currentDayOfStart) {
            authGroup.currentLimitPerDay = 1
            authGroup.currentLimitPerMin = 1
            authGroup.lastCallTime = currentTime.getTime()

            await this.ctx.database.upsert('chathub_auth_group', [authGroup])

            return
        }

        // Check to see if it's been more than a minute since the last call

        if (currentTime.getTime() - authGroup.lastCallTime >= 60000) {
            // clear

            authGroup.currentLimitPerDay += 1
            authGroup.currentLimitPerMin = 1
            authGroup.lastCallTime = currentTime.getTime()

            await this.ctx.database.upsert('chathub_auth_group', [authGroup])
        }

        authGroup.currentLimitPerDay += 1
        authGroup.currentLimitPerMin += 1

        await this.ctx.database.upsert('chathub_auth_group', [authGroup])
    }

    async addUserToGroup(user: ChatHubAuthUser, groupName: string) {
        const group = (
            await this.ctx.database.get('chathub_auth_group', {
                name: groupName
            })
        )?.[0]

        if (group == null) {
            throw new ChatHubError(ChatHubErrorCode.AUTH_GROUP_NOT_FOUND)
        }

        await this.ctx.database.upsert('chathub_auth_joined_user', [
            {
                userId: user.userId,
                groupId: group.id,
                groupName: group.name
            }
        ])
    }

    private async _initAuthGroup() {
        // init guest group

        let guestGroup = (
            await this.ctx.database.get('chathub_auth_group', {
                name: 'guest'
            })
        )?.[0]

        if (guestGroup == null) {
            guestGroup = {
                name: 'guest',
                priority: 0,

                limitPerMin: 10,
                limitPerDay: 2000,

                // 1000 token / 0.3
                constPerToken: 0.3,
                id: undefined,
                supportModels: undefined
            }

            await this.ctx.database.upsert('chathub_auth_group', [guestGroup])
        }

        let userGroup = (
            await this.ctx.database.get('chathub_auth_group', { name: 'user' })
        )?.[0]

        if (userGroup == null) {
            userGroup = {
                name: 'user',
                priority: 0,
                limitPerMin: 1000,
                limitPerDay: 200000,

                // 1000 token / 0.01
                constPerToken: 0.01,
                id: undefined,
                supportModels: undefined
            }
        }

        let adminGroup = (
            await this.ctx.database.get('chathub_auth_group', {
                name: 'admin'
            })
        )?.[0]

        if (adminGroup == null) {
            adminGroup = {
                name: 'admin',
                priority: 0,
                limitPerMin: 10000,
                limitPerDay: 20000000,

                // 1000 token / 0.001
                constPerToken: 0.001,
                id: undefined,
                supportModels: undefined
            }

            await this.ctx.database.upsert('chathub_auth_group', [adminGroup])
        }
    }

    private async _defineDatabase() {
        const ctx = this.ctx

        ctx.database.extend(
            'chathub_auth_user',
            {
                userId: {
                    type: 'string'
                },
                balance: {
                    type: 'decimal'
                },
                authType: {
                    type: 'char',
                    length: 50
                }
            },
            {
                autoInc: false,
                primary: 'userId',
                unique: ['userId']
            }
        )

        ctx.database.extend(
            'chathub_auth_joined_user',
            {
                userId: 'string',
                groupId: 'integer',
                groupName: 'string',
                id: 'integer'
            },
            {
                autoInc: true,
                primary: 'id',
                unique: ['id']
            }
        )

        ctx.database.extend(
            'chathub_auth_group',
            {
                limitPerDay: {
                    type: 'integer',
                    nullable: false
                },
                limitPerMin: {
                    type: 'integer',
                    nullable: false
                },
                lastCallTime: {
                    type: 'integer',
                    nullable: true
                },
                currentLimitPerDay: {
                    type: 'integer',
                    nullable: true
                },
                currentLimitPerMin: {
                    type: 'integer',
                    nullable: true
                },
                supportModels: {
                    type: 'json',
                    nullable: true
                },
                priority: {
                    type: 'integer',
                    nullable: false,
                    initial: 0
                },
                platform: {
                    type: 'char',
                    length: 255,
                    nullable: true
                },
                constPerToken: {
                    type: 'integer'
                },
                name: {
                    type: 'char',
                    length: 255
                },
                id: {
                    type: 'integer'
                }
            },
            {
                autoInc: true,
                primary: 'id',
                unique: ['id', 'name']
            }
        )

        await this._initAuthGroup()
    }
}
