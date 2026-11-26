begin;

create table if not exists public.tutor_subscription_access (
    subscription_id text primary key,
    email text,
    ended_at timestamptz
);

create index if not exists tutor_active_subscription_email_idx
    on public.tutor_subscription_access (email) where ended_at is null;

alter table public.tutor_subscription_access enable row level security;
revoke all on public.tutor_subscription_access from public, anon, authenticated, service_role;

create or replace function public.record_tutor_subscription_access(
    p_subscription_id text,
    p_email text default null,
    p_source text default null,
    p_contact_id text default null,
    p_ended_at timestamptz default null
) returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
    buyer_email text;
begin
    if p_subscription_id is null or p_subscription_id !~ '^sub_[A-Za-z0-9]{1,124}$' then
        raise exception 'Invalid Stripe subscription ID';
    end if;

    if p_ended_at is not null then
        if p_ended_at > clock_timestamp() then
            raise exception 'Subscription end is in the future';
        end if;
        -- Keep an end even when Stripe delivers it before the checkout webhook arrives.
        insert into public.tutor_subscription_access as entitlement (subscription_id, ended_at)
        values (p_subscription_id, p_ended_at)
        on conflict (subscription_id) do update
            set ended_at = coalesce(entitlement.ended_at, excluded.ended_at)
        returning entitlement.email into buyer_email;
    else
        if p_email is null or p_email = '' or p_email <> lower(btrim(p_email))
            or p_source is null or p_source not in ('stripe', 'promo') then
            raise exception 'Invalid subscription grant';
        end if;
        insert into public.tutor_subscription_access as entitlement (subscription_id, email)
        values (p_subscription_id, p_email)
        on conflict (subscription_id) do update set email = excluded.email
            where entitlement.ended_at is null and entitlement.email = excluded.email
        returning entitlement.email into buyer_email;
    end if;

    if buyer_email is null then
        return false;
    end if;

    -- Serialize the authorization projection for this email when multiple subscriptions change.
    perform pg_advisory_xact_lock(hashtextextended(buyer_email, 0));
    if p_ended_at is null then
        insert into public.authorized_users as access_row (email, authorized, source, flow_contact_id, updated_at)
        values (buyer_email, true, p_source || '_subscription', p_contact_id, clock_timestamp())
        on conflict (email) do update set
            authorized = true,
            source = excluded.source,
            flow_contact_id = excluded.flow_contact_id,
            updated_at = excluded.updated_at
        -- An existing manual or legacy entitlement remains independent of this subscription.
        where access_row.authorized is not true
            or access_row.source in ('stripe_subscription', 'promo_subscription');
    else
        update public.authorized_users
        set authorized = false, updated_at = clock_timestamp()
        where email = buyer_email and authorized is true
          and source in ('stripe_subscription', 'promo_subscription')
          and not exists (
              select 1 from public.tutor_subscription_access
              where email = buyer_email and ended_at is null
          );
    end if;
    return true;
end;
$$;

revoke all on function public.record_tutor_subscription_access(text, text, text, text, timestamptz)
    from public, anon, authenticated;
grant execute on function public.record_tutor_subscription_access(text, text, text, text, timestamptz)
    to service_role;

commit;
