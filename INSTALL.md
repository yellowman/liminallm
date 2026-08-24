# Install

Requires PostgreSQL 14+ with the `vector` and `citext` extensions, Redis, and
Python 3.10+.

Pick one path: Docker on Linux, Linux without Docker, or OpenBSD.

---

## Docker (Linux)

```sh
git clone https://github.com/yellowman/liminallm.git && cd liminallm
cp .env.example .env
```

Edit `.env` and set at least:

```sh
POSTGRES_PASSWORD=...
REDIS_PASSWORD=...
OPENAI_API_KEY=sk-...     # or the key for whichever provider you use
```

Start it:

```sh
docker compose up -d
```

Schema is applied by the `migrate` service before the app starts. Create the
first admin:

```sh
docker compose exec app python scripts/bootstrap_admin.py \
  --email you@example.com --password 'YourPassword123!'
```

Open <http://localhost:8000>.

---

## Linux (no Docker)

```sh
# Debian/Ubuntu
apt install python3 python3-pip postgresql postgresql-16-pgvector redis-server

# Fedora/RHEL
dnf install python3 python3-pip postgresql-server pgvector redis
```

Database:

```sh
sudo -u postgres createuser -P liminallm
sudo -u postgres createdb -O liminallm liminallm
sudo -u postgres psql -d liminallm -c 'CREATE EXTENSION vector; CREATE EXTENSION citext;'
```

Application:

```sh
git clone https://github.com/yellowman/liminallm.git /opt/liminallm && cd /opt/liminallm
pip install -e .
install -d -o "$USER" /srv/liminallm

export DATABASE_URL='postgresql://liminallm:PASSWORD@localhost/liminallm'
./scripts/migrate.sh
python scripts/bootstrap_admin.py --email you@example.com --password 'YourPassword123!'
python -m uvicorn liminallm.app:app --host 127.0.0.1 --port 8000
```

To run it as a service, create `/etc/systemd/system/liminallm.service`:

```ini
[Unit]
After=network.target postgresql.service redis.service

[Service]
User=liminallm
WorkingDirectory=/opt/liminallm
EnvironmentFile=/etc/liminallm/env
ExecStart=/usr/bin/python3 -m uvicorn liminallm.app:app --host 127.0.0.1 --port 8000 --workers 4
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Put `DATABASE_URL` and any provider API key in `/etc/liminallm/env`
(`chmod 640`), then `systemctl enable --now liminallm`.

---

## OpenBSD

```sh
pkg_add python3 postgresql-server redis pgvector
useradd -s /sbin/nologin -d /var/liminallm -c "LiminalLM" _liminallm
install -d -o _liminallm -g _liminallm -m 750 /var/liminallm
install -d -o root -g _liminallm -m 750 /etc/liminallm
```

Database:

```sh
rcctl enable postgresql && rcctl start postgresql
su - _postgresql -c 'createuser -P liminallm'
su - _postgresql -c 'createdb -O liminallm liminallm'
psql -U liminallm -d liminallm -c 'CREATE EXTENSION vector; CREATE EXTENSION citext;'
rcctl enable redis && rcctl start redis
```

Application:

```sh
cd /var/liminallm
git clone https://github.com/yellowman/liminallm.git .
pip3 install -e .

cp deploy/openbsd/env.example /etc/liminallm/env
chmod 640 /etc/liminallm/env && chown root:_liminallm /etc/liminallm/env
```

Set `DATABASE_URL` and any provider API key in `/etc/liminallm/env`, then:

```sh
export DATABASE_URL='postgresql://liminallm:PASSWORD@localhost/liminallm'
./scripts/migrate.sh
python3 scripts/bootstrap_admin.py --email you@example.com --password 'YourPassword123!'

install -m 755 deploy/openbsd/rc.d/liminallm /etc/rc.d/liminallm
rcctl enable liminallm && rcctl start liminallm
```

For httpd/relayd in front, use `deploy/openbsd/httpd.conf` and
`deploy/openbsd/relayd.conf`.

---

## Configuration

Only these are environment variables:

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | yes | Postgres connection string |
| `SHARED_FS_ROOT` | no | Where the data lives on this machine (default `/srv/liminallm`) |
| `EMBEDDING_VECTOR_DIM` | no | Embedding width, fixed when the schema is first created (default 1536) |
| `<PROVIDER>_API_KEY` | no | `OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, … used when the matching admin setting is blank |
| `INSTANCE_SETTINGS_JSON` | no | JSON object seeding managed settings on first boot only |

Everything else — model, provider, Redis URL, SMTP, OAuth, rate limits — is set
in the admin console at `/` after signing in as an admin. `JWT_SECRET` is
generated on first boot if unset.

`EMBEDDING_VECTOR_DIM` sets the width of the vector columns, and that width is
fixed when the schema is first created. The width comes from the
`CREATE TABLE IF NOT EXISTS` declaration that creates each vector column, so
re-running `scripts/migrate.sh` cannot change it: the table already exists, the
declaration is skipped, and the column keeps the type it was created with.

To change the width on an existing database, you must drop and recreate the
vector columns, then re-embed all stored content. The application compares the
column width against the configured encoder at startup, and refuses to start
when the two differ.

---

## Verify

```sh
curl http://127.0.0.1:8000/healthz
```
