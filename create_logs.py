import random

# =============== CONFIG ===============
NUM_LINES = 200_000   # total de linhas no arquivo final
# random.seed(42)     # opcional

# Probabilidades (pesos) dos tipos de bloco emitidos a cada passo
WEIGHTS = {
    "causal": 0.40,     # emitir um trecho de cadeia causal
    "spurious": 0.15,   # emitir um par espúrio (ordem invertida)
    "common": 0.15,     # emitir uma causa comum + alguns efeitos
    "noise": 0.15,      # emitir 1..MAX_NOISE_PER_BURST linhas de ruído
    "single": 0.15,     # emitir 1 ação avulsa
}

# Tamanhos típicos de “blocos” (não são sequências inteiras; apenas bursts)
CAUSAL_MIN, CAUSAL_MAX = 2, 3       # quantas etapas da cadeia causal emitir
MAX_NOISE_PER_BURST = 3             # ruído por burst

# =============== DADOS ===============
# Ações normais (sem tokens tipo [ERROR], etc.)
actions = [
    "Cloning repository", "Checking out branch", "Installing dependencies", "Verifying dependency integrity",
    "Configuring build environment", "Building CXX object", "Building Java classes", "Compiling TypeScript files",
    "Linking CXX executable", "Linking shared libraries", "Running unit tests", "Running integration tests",
    "Generating code coverage report", "Reporting test results", "Checking code style", "Running linter checks",
    "Analyzing static code", "Creating version header", "Compressing log files", "Uploading logs to S3",
    "Triggering webhook", "Sending Slack notification", "Deploying to staging environment", "Deploying to production",
    "Validating deployment", "Running smoke tests", "Starting CI job", "Finalizing CI job",
    "Pushing Docker image", "Pulling Docker base image", "Tagging Docker image", "Cleaning Docker cache",
    "Encrypting artifacts", "Decrypting configuration files", "Backing up database", "Restoring database",
    "Migrating database schema", "Seeding database", "Checking database connectivity", "Restarting database service",
    "Verifying checksums", "Running security scan", "Patching vulnerabilities", "Reviewing dependencies",
    "Archiving build artifacts", "Publishing artifacts to repository", "Generating documentation with Doxygen",
    "Converting markdown to HTML", "Publishing site", "Notifying release manager", "Syncing with GitHub",
    "Merging pull request", "Rebasing branch", "Creating release notes", "Signing release packages",
    "Verifying digital signatures", "Pushing changes to remote", "Creating git tag", "Verifying git tag signature",
    "Running regression tests", "Running performance tests", "Measuring memory usage", "Analyzing CPU usage",
    "Formatting source code", "Optimizing assets", "Uploading sourcemaps", "Tracking build metrics",
    "Analyzing historical trends", "Creating Jira ticket", "Logging build statistics", "Sending report via email",
    "Scanning for secrets in code", "Checking disk space", "Checking available memory", "Monitoring build agents",
    "Syncing mirrors", "Rebooting build server", "Rebuilding failed jobs", "Creating new pipeline configuration",
    "Triggering downstream jobs", "Validating Kubernetes manifests", "Deploying Helm charts",
    "Verifying service health checks", "Logging health status", "Generating system diagnostics",
    "Uploading crash reports", "Restarting failed containers", "Rebalancing workloads"
]
ACTIONS_SET = set(actions)

# Ruído (sem prefixos tipo "[NOISE]")
noise_lines = [
    "Linker returned non-zero exit code",
    "Deprecated function used",
    "Entering compilation loop",
    "Out of memory during linking stage",
    "Dependency chain resolved",
    "dmesg: CPU soft lockup detected on core 3",
    "gc: collecting generation 2 ... done",
    "java.lang.NullPointerException at com.example.Main:142",
    "WARN retrying connection to mirror (attempt 3/5)",
    "TRACE socket recv timeout, backing off 200ms",
    "^[[0;31mANSI color spill^[[0m",
    "###### random boundary #####",
]

# Cadeias causais realistas (mantidas; cadeias inválidas são filtradas)
causal_chains_realistic = [
    ["Cloning repository", "Checking out branch", "Installing dependencies"],
    ["Installing dependencies", "Configuring build environment", "Building CXX object"],
    ["Building CXX object", "Linking CXX executable", "Running unit tests"],
    ["Running unit tests", "Generating code coverage report", "Reporting test results"],
    ["Running integration tests", "Validating deployment", "Running smoke tests"],
    ["Migrating database schema", "Seeding database", "Checking database connectivity"],
    ["Backing up database", "Restoring database", "Verifying checksums"],
    ["Deploying to staging environment", "Deploying to production", "Validating deployment"],
    ["Packaging project into tar.gz", "Uploading logs to S3", "Triggering webhook"],  # será ignorada
    ["Starting CI job", "Finalizing CI job", "Notifying release manager"]
]
CAUSAL_RULES = [chain for chain in causal_chains_realistic if all(a in ACTIONS_SET for a in chain)]

# Causa comum simples
COMMON_CAUSE = "Cloning repository"
COMMON_EFFECTS = [a for a in actions if ("Installing" in a) or ("Checking out" in a) or ("Building" in a)]
COMMON_EFFECTS = COMMON_EFFECTS[:4]

# Pares invertidos espúrios
REVERSED_PAIRS = [
    ("Running integration tests", "Deploying to production"),
    ("Uploading logs to S3", "Triggering webhook"),
    ("Generating documentation with Doxygen", "Publishing site"),
    ("Migrating database schema", "Backing up database"),
]
REVERSED_PAIRS = [(b, a) for (a, b) in REVERSED_PAIRS if a in ACTIONS_SET and b in ACTIONS_SET]

# =============== EMISSORES DE "BLOCOS" ===============
def emit_causal_burst():
    """Emite 2–3 passos consecutivos de uma cadeia causal realista."""
    if not CAUSAL_RULES:
        return [random.choice(actions)]
    chain = random.choice(CAUSAL_RULES)
    k = random.randint(CAUSAL_MIN, CAUSAL_MAX)
    k = min(k, len(chain))
    return chain[:k]

def emit_spurious_pair():
    """Emite um par espúrio na ordem invertida (b, a)."""
    if not REVERSED_PAIRS:
        return [random.choice(actions)]
    a, b = random.choice(REVERSED_PAIRS)  # armazenamos como (b, a) já invertido acima
    return [a, b]

def emit_common_burst():
    """Emite a causa comum + alguns efeitos (subset aleatório)."""
    if COMMON_CAUSE not in ACTIONS_SET:
        return [random.choice(actions)]
    effects = [e for e in COMMON_EFFECTS if random.random() < 0.8]  # 80% de chance para cada
    if not effects:
        effects = [random.choice(COMMON_EFFECTS)]
    return [COMMON_CAUSE] + effects

def emit_noise_burst():
    """Emite 1..MAX_NOISE_PER_BURST linhas de ruído."""
    k = random.randint(1, MAX_NOISE_PER_BURST)
    k = min(k, len(noise_lines))
    return random.sample(noise_lines, k)

def emit_single_action():
    return [random.choice(actions)]

EMITTERS = [
    ("causal",  emit_causal_burst),
    ("spurious", emit_spurious_pair),
    ("common", emit_common_burst),
    ("noise",  emit_noise_burst),
    ("single", emit_single_action),
]

# Normaliza pesos
labels, funcs = zip(*EMITTERS)
weights = [WEIGHTS[l] for l in labels]
s = sum(weights)
weights = [w / s for w in weights]

# =============== GERAÇÃO STREAM ===============
lines_out = []
while len(lines_out) < NUM_LINES:
    fn = random.choices(funcs, weights=weights, k=1)[0]
    chunk = fn()
    # Evita ultrapassar NUM_LINES
    remaining = NUM_LINES - len(lines_out)
    if len(chunk) > remaining:
        chunk = chunk[:remaining]
    lines_out.extend(chunk)

# =============== I/O ===============
with open("synthetic_sequences.txt", "w") as f:
    f.write("\n".join(lines_out) + "\n")
