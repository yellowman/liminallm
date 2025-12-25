# LiminalLM OpenBSD Deployment

This directory contains configuration files for deploying LiminalLM on OpenBSD.

## Prerequisites

Install required packages:

```sh
pkg_add python3 py3-pip postgresql-server redis pgvector
```

## Setup

### 1. Create service user

```sh
useradd -s /sbin/nologin -d /var/liminallm -c "LiminalLM Service" _liminallm
```

### 2. Install LiminalLM

```sh
# Create directories
install -d -o _liminallm -g _liminallm -m 750 /var/liminallm
install -d -o root -g _liminallm -m 750 /etc/liminallm

# Install Python package
pip3 install /path/to/liminallm

# Or install from source
cd /var/liminallm
git clone https://github.com/your-org/liminallm.git .
pip3 install -e .
```

### 3. Configure PostgreSQL

```sh
# Initialize and start PostgreSQL
rcctl enable postgresql
rcctl start postgresql

# Create database and user
su - _postgresql
createuser -P liminallm
createdb -O liminallm liminallm

# Enable pgvector extension
psql -d liminallm -c "CREATE EXTENSION vector;"
psql -d liminallm -c "CREATE EXTENSION citext;"

# Run migrations
psql -d liminallm -f /var/liminallm/sql/000_base.sql
psql -d liminallm -f /var/liminallm/sql/001_artifacts.sql
psql -d liminallm -f /var/liminallm/sql/002_knowledge.sql
psql -d liminallm -f /var/liminallm/sql/003_preferences.sql
psql -d liminallm -f /var/liminallm/sql/004_runtime_config.sql
```

### 4. Configure Redis

```sh
rcctl enable redis
rcctl start redis
```

### 5. Configure environment

```sh
# Copy and edit environment file
cp env.example /etc/liminallm/env
chmod 640 /etc/liminallm/env
chown root:_liminallm /etc/liminallm/env
vi /etc/liminallm/env
```

### 6. Install rc.d script

```sh
cp rc.d/liminallm /etc/rc.d/
chmod 755 /etc/rc.d/liminallm
rcctl enable liminallm
rcctl start liminallm
```

### 7. Configure reverse proxy

**Option A: Using relayd (recommended for WebSocket support)**

```sh
# Generate TLS certificate
acme-client -v liminallm.example.com

# Copy relayd configuration
cp relayd.conf /etc/relayd.conf
vi /etc/relayd.conf  # Adjust domain and paths

# Enable and start relayd
rcctl enable relayd
rcctl start relayd
```

**Option B: Using httpd (simpler, limited WebSocket)**

```sh
cp httpd.conf /etc/httpd.conf
vi /etc/httpd.conf  # Adjust domain and paths

rcctl enable httpd
rcctl start httpd
```

## Management

```sh
# Check status
rcctl check liminallm

# View logs
tail -f /var/log/liminallm/app.log

# Restart service
rcctl restart liminallm

# Reload configuration (relayd)
rcctl reload relayd
```

## Firewall (pf.conf)

Add to `/etc/pf.conf`:

```
# LiminalLM
pass in on egress proto tcp from any to (egress) port { 80, 443 }
```

Then reload: `pfctl -f /etc/pf.conf`

## TLS Certificates

Using acme-client for Let's Encrypt:

```sh
# Add to /etc/acme-client.conf
domain liminallm.example.com {
    domain key "/etc/ssl/private/liminallm.key"
    domain certificate "/etc/ssl/liminallm.crt"
    domain full chain certificate "/etc/ssl/liminallm.fullchain.pem"
    sign with letsencrypt
}

# Request certificate
acme-client -v liminallm.example.com

# Add to crontab for renewal
0 0 * * * acme-client liminallm.example.com && rcctl reload relayd
```

## Troubleshooting

**Service won't start:**
```sh
# Check environment file syntax
ksh -n /etc/liminallm/env

# Check permissions
ls -la /var/liminallm /etc/liminallm

# Run manually for debugging
su -s /bin/ksh _liminallm -c ". /etc/liminallm/env && python3 -m uvicorn liminallm.app:app --host 127.0.0.1 --port 8000"
```

**Database connection issues:**
```sh
# Test connection
psql -U liminallm -d liminallm -c "SELECT 1"

# Check pg_hba.conf
cat /var/postgresql/data/pg_hba.conf
```

**Redis connection issues:**
```sh
redis-cli ping
```
