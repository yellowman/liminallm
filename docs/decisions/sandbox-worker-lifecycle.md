# why the tool worker lifecycle is this strict

SPEC §18.3 states the invariants: a worker per attempt in its own process
group, revocation before kill, confirmed reaping, capability checks at the
parent, a linearized commit guard, an ordered operation ledger. Each rule
closed a hole. This file records the holes.

## a python thread cannot be killed

Tool handlers used to run on a pool thread. `future.cancel()` returns false
for anything already running, so a node timeout cancelled the coroutine and
the handler carried on beside its own retry. That is why the unit of tool
execution is a spawned worker process per attempt — a process can be
killed; a thread can only be asked.

## the setsid window

The child calls `setsid` after `start()` returns, so until it has,
`getpgid(child)` answers with the *parent's* group — and a `killpg` in that
window SIGKILLs the api server and everything sharing its group. The child
therefore *earns* the group: it sends a ready handshake once `setsid` has
actually happened, carrying the pgid it landed in, and only a pgid equal to
its own pid promotes the registration from single-pid to group. The kill
path re-checks the same thing, because the cost of the two disagreeing is
the whole process group.

## pid reuse

A pid outlives its process only as a number, and the kernel reuses numbers,
so a registration left behind after a child is reaped is a standing licence
to signal whoever inherits that number — redeemed at teardown, against a
stranger. Registration hands back the means to undo it, and the normal exit
path uses it.

## revocation before kill, and the commit guard

Killing first lets an effect that has already passed its check start
against a tree the parent has torn down. And `check(); COMMIT` leaves a
window where revocation lands between the two, so the timeout path can
report revocation complete while an authorized commit is still in flight.
`commit_guard` and `revoke` contend on the same per-invocation lock,
leaving only two histories: the commit runs and revocation waits for it, or
revocation completes and the commit is refused.

The guard goes around the *mutation*, never around the call that leads to
it: between "the handler was entered" and "the row exists" there is a
window, and a retry landing inside it either duplicates the write or skips
it depending on which side of the boundary the guard sat. (The same shape
appeared twice more in the account-erasure work — see docs/ISSUES.md,
tranches 2G.4 "the write side" and "the claim is a write".)

## why the ledger is ordered, not content-addressed

A key of `execution_id + operation_name` collides two legitimate identical
calls into one, and misses a retry whose model-written payload differs by a
byte. An ordered entry `(operation_seq, capability, payload_hash, state,
result)` lets a replacement worker replay its control flow and stamp each
request with its position: a committed step at that position returns its
stored result instead of happening twice; a divergent durable retry is
refused; a read simply runs again. A step still `pending` when its attempt
died becomes `unknown`, not `failed` — nothing left can say whether it
landed, and a durable `unknown` is refused rather than repeated.

The payload hash of a publication covers the *bytes* of each file, not only
its name: a retry runs the model's code again, and the same code writing
`result.csv` down a different branch produces the same name over different
content. Replaying on the name would leave attempt one's file in the user's
area while attempt two's answer describes what it computed, with nothing
reporting the disagreement.

## why the worker is confined before any body runs

"The broker is its only channel" is a property of the process, not of the
protocol. A spawned child inherits the service's environment, filesystem
view and network namespace, so a worker that merely *intends* to reach the
world through the broker still holds `DATABASE_URL`, `open('/etc/passwd')`
and an outbound socket. One bug in a tool body is the difference, and
containing that bug is what the process is for.

The environment is inherited at process start and lives in memory, so
re-rooting does nothing to it; the network is removed structurally, because
refusing to import `socket` missed `_socket`, and any already-loaded module
can hand out the same primitive. The service uid owns every user's files,
so unix permissions are the wrong instrument and `os.chdir()` is not
confinement at all.

## why privileged is a conjunction read from the row

`/v1/artifacts` is open to any authenticated user and the tool schema
permits additional properties, so an ordinary user could author
`privileged: true` and an admin invoking it would be handed the privileged
sandbox for someone else's definition. Ownership is read from the artifact
row, never from a field inside `schema` — a spec naming its own owner is
quoting itself.

Two related resolution rules earned the same way:

- Caching a private tool into the process-wide registry made one user's
  private definition resolvable for every later request in that process —
  resolution is per request.
- Handing the engine a bare schema and letting it resolve the name again is
  a substitution: names carry no uniqueness constraint, so the second
  lookup can return a different row, including one that declares
  `privileged: true` where the authorized row did not. An invocation of an
  id stays bound to that id.

## invocation state travels with the work

Hot reload replaces the engine while in-flight work finishes. A process
global would have an old attempt asking the new engine about an execution
it never opened — a refusal indistinguishable from a real revocation. The
registry of live executions belongs to the engine, and each entry is opened
once and closed once on every terminal path, revocation included.

The check also follows the work, not the thread that started it: parallel
reads run in a nested pool, the bound invocation is thread-local, and an
unbound thread reads as the api path and passes every check — so the
invocation is re-applied in every worker, on every call, reads included. A
list of "write methods" would be a guess about which calls matter.

## why findings withdraw capabilities at the capability itself

A turn that has read a possible injection loses `run_python`, `web_fetch`,
and `web_search` for the rest of the turn. The findings live on the
invocation, parent-side, and the refusal happens at the capability — not
merely inside the round that usually carries it — because the worker is the
untrusted side: "it asks through the round" describes the intended
protocol, not a constraint on a compromised one, and a worker that has just
read a hostile page can ask for `web.fetch` directly. The process that read
"ignore your rules and run this" is the last one that should be asked
whether the rule still applies. Withdrawal covers the same round, which is
why anything that can taint runs in order while pure reads may fan out.
