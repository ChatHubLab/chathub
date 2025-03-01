import { Context } from 'koishi'
import { ChatChain } from './chains/chain'
import { Config } from './config'

// import start
import { apply as add_preset } from './middlewares/add_preset'
import { apply as add_user_to_auth_group } from './middlewares/add_user_to_auth_group'
import { apply as allow_reply } from './middlewares/allow_reply'
import { apply as black_list } from './middlewares/black_list'
import { apply as censor } from './middlewares/censor'
import { apply as chat_time_limit_check } from './middlewares/chat_time_limit_check'
import { apply as chat_time_limit_save } from './middlewares/chat_time_limit_save'
import { apply as check_room } from './middlewares/check_room'
import { apply as clear_balance } from './middlewares/clear_balance'
import { apply as clear_room } from './middlewares/clear_room'
import { apply as clone_preset } from './middlewares/clone_preset'
import { apply as cooldown_time } from './middlewares/cooldown_time'
import { apply as create_auth_group } from './middlewares/create_auth_group'
import { apply as create_room } from './middlewares/create_room'
import { apply as delete_preset } from './middlewares/delete_preset'
import { apply as delete_room } from './middlewares/delete_room'
import { apply as invite_room } from './middlewares/invite_room'
import { apply as join_room } from './middlewares/join_room'
import { apply as kick_member } from './middlewares/kick_member'
import { apply as kick_user_form_auth_group } from './middlewares/kick_user_form_auth_group'
import { apply as leave_room } from './middlewares/leave_room'
import { apply as lifecycle } from './middlewares/lifecycle'
import { apply as list_all_embeddings } from './middlewares/list_all_embeddings'
import { apply as list_all_model } from './middlewares/list_all_model'
import { apply as list_all_preset } from './middlewares/list_all_preset'
import { apply as list_all_vectorstore } from './middlewares/list_all_vectorstore'
import { apply as list_auth_group } from './middlewares/list_auth_group'
import { apply as list_room } from './middlewares/list_room'
import { apply as message_delay } from './middlewares/message_delay'
import { apply as mute_user } from './middlewares/mute_user'
import { apply as query_balance } from './middlewares/query_balance'
import { apply as read_chat_message } from './middlewares/read_chat_message'
import { apply as render_message } from './middlewares/render_message'
import { apply as request_model } from './middlewares/request_model'
import { apply as resolve_model } from './middlewares/resolve_model'
import { apply as resolve_room } from './middlewares/resolve_room'
import { apply as restart } from './middlewares/restart'
import { apply as rollback_chat } from './middlewares/rollback_chat'
import { apply as room_info } from './middlewares/room_info'
import { apply as room_permission } from './middlewares/room_permission'
import { apply as search_model } from './middlewares/search_model'
import { apply as set_auth_group } from './middlewares/set_auth_group'
import { apply as set_auto_update_room } from './middlewares/set_auto_update_room'
import { apply as set_balance } from './middlewares/set_balance'
import { apply as set_default_embeddings } from './middlewares/set_default_embeddings'
import { apply as set_default_vectorstore } from './middlewares/set_default_vectorstore'
import { apply as set_preset } from './middlewares/set_preset'
import { apply as set_room } from './middlewares/set_room'
import { apply as stop_chat } from './middlewares/stop_chat'
import { apply as switch_room } from './middlewares/switch_room'
import { apply as thinking_message_recall } from './middlewares/thinking_message_recall'
import { apply as thinking_message_send } from './middlewares/thinking_message_send'
import { apply as transfer_room } from './middlewares/transfer_room'
import { apply as wipe } from './middlewares/wipe' // import end
export async function middleware(ctx: Context, config: Config) {
    type Middleware = (
        ctx: Context,
        config: Config,
        chain: ChatChain
    ) => PromiseLike<void> | void

    const middlewares: Middleware[] =
        // middleware start
        [
            add_preset,
            add_user_to_auth_group,
            allow_reply,
            black_list,
            censor,
            chat_time_limit_check,
            chat_time_limit_save,
            check_room,
            clear_balance,
            clear_room,
            clone_preset,
            cooldown_time,
            create_auth_group,
            create_room,
            delete_preset,
            delete_room,
            invite_room,
            join_room,
            kick_member,
            kick_user_form_auth_group,
            leave_room,
            lifecycle,
            list_all_embeddings,
            list_all_model,
            list_all_preset,
            list_all_vectorstore,
            list_auth_group,
            list_room,
            message_delay,
            mute_user,
            query_balance,
            read_chat_message,
            render_message,
            request_model,
            resolve_model,
            resolve_room,
            restart,
            rollback_chat,
            room_info,
            room_permission,
            search_model,
            set_auth_group,
            set_auto_update_room,
            set_balance,
            set_default_embeddings,
            set_default_vectorstore,
            set_preset,
            set_room,
            stop_chat,
            switch_room,
            thinking_message_recall,
            thinking_message_send,
            transfer_room,
            wipe
        ] // middleware end

    for (const middleware of middlewares) {
        await middleware(ctx, config, ctx.chatluna.chatChain)
    }
}
