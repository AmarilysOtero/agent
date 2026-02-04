# Phase 1 Corrected Implementation — RLM Request Parameter Support

**Date**: February 4, 2026

**Status**: ✅ IMPLEMENTATION ADJUSTED TO MATCH SPEC

---

## Correction Made

**Original Issue**: Phase 1 was implemented as a global environment variable only, not as a per-execution UI toggle.

**Spec Requirement**: `RLM_ENABLED` should be toggled from the Orchestration side panel per execution.

**Solution**: Added request parameter support so Front-end can pass `rlm_enabled` per-request.

---

## Changes

### 1. Backend API Request Model

**File**: `src/news_reporter/routers/workflows.py`

**Added Field**:

```python
class WorkflowRequest(BaseModel):
    """Request model for workflow execution"""
    goal: str
    graph_path: Optional[str] = None
    workflow_definition: Optional[Dict[str, Any]] = None
    workflow_id: Optional[str] = None
    use_graph: bool = True
    checkpoint_dir: Optional[str] = None
    rlm_enabled: Optional[bool] = Field(
        None,
        description="Enable RLM answering flow (overrides config if provided)"
    )
```

**Override Logic**:

```python
@router.post("/execute")
async def execute_workflow(request: WorkflowRequest) -> WorkflowResponse:
    config = Settings.load()

    # Override RLM setting if provided in request (per-execution control)
    if request.rlm_enabled is not None:
        config.rlm_enabled = request.rlm_enabled

    # ... rest of execution
```

### 2. Frontend Type Definition

**File**: `src/types/workflow.ts`

**Updated Interface**:

```typescript
export interface WorkflowRequest {
	goal: string;
	graph_path?: string;
	workflow_definition?: WorkflowDefinition;
	workflow_id?: string;
	use_graph?: boolean;
	checkpoint_dir?: string;
	rlm_enabled?: boolean; // NEW: Per-execution RLM toggle
}
```

### 3. Implementation Plan Updated

**File**: `RLM_IMPLEMENTATION_PLAN.md`

**UI Enablement Section**: Clarified the request flow:

```
Front-End (Orchestration Panel)
  ↓ (rlm_enabled=true/false in WorkflowRequest)
→ Backend API (/api/v1/workflows/execute)
  ↓ (request.rlm_enabled overrides config.rlm_enabled if provided)
→ Config (falls back to RLM_ENABLED env var if request param is None)
  ↓
→ Workflow Factory (routes to RLM or default branch)
```

---

## How It Works

### Execution Flow

1. **User Action**: Toggle RLM in Orchestration panel (Advanced/Experimental options)
2. **Front-End**: Passes `rlm_enabled=true` or `false` in POST request to `/api/v1/workflows/execute`
3. **Backend**: Receives `WorkflowRequest` with `rlm_enabled` parameter
4. **Config Override**: If `request.rlm_enabled` is not None, override `config.rlm_enabled`
5. **Fallback**: If `request.rlm_enabled` is None, use `RLM_ENABLED` env var
6. **Routing**: Workflow checks `config.rlm_enabled` and routes accordingly

### Priority Order (Highest to Lowest)

1. **Request Parameter** (`rlm_enabled` in WorkflowRequest)
2. **Environment Variable** (`RLM_ENABLED` in .env)
3. **Default** (false)

---

## Configuration

### Environment Variables (Global Defaults)

```bash
# Global default (used when request doesn't specify)
RLM_ENABLED=false
```

### Request Parameter (Per-Execution Override)

**Orchestration Panel sends:**

```json
{
	"goal": "Tell me about...",
	"use_graph": false,
	"rlm_enabled": true
}
```

---

## Testing

### Test 1: Toggle in Orchestration Panel

1. Open Orchestration tab → Advanced/Experimental options
2. Toggle RLM ON
3. Submit a query
4. Check logs for: "RLM branch selected"
5. Toggle RLM OFF
6. Submit same query
7. Check logs for: "Default sequential branch selected (RLM not enabled)"

### Test 2: Request Parameter Override

**Via REST API**:

```bash
curl -X POST http://localhost:8787/api/v1/workflows/execute \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Test query",
    "use_graph": false,
    "rlm_enabled": true
  }'
```

Expected behavior: RLM branch selected, even if `RLM_ENABLED=false` in .env

### Test 3: Fallback to Config

Send request without `rlm_enabled`:

```bash
curl -X POST http://localhost:8787/api/v1/workflows/execute \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Test query",
    "use_graph": false
  }'
```

Expected behavior: Uses `RLM_ENABLED` env var (or false if not set)

---

## Files Modified

| File                                              | Change                                                       |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `src/news_reporter/routers/workflows.py`          | Added `rlm_enabled` to `WorkflowRequest` + override logic    |
| `src/news_reporter/config.py`                     | (Already has `rlm_enabled` from Phase 1)                     |
| `src/news_reporter/workflows/workflow_factory.py` | (Already has routing logic from Phase 1)                     |
| `src/types/workflow.ts` (Front-end)               | Added `rlm_enabled?: boolean` to `WorkflowRequest` interface |
| `RLM_IMPLEMENTATION_PLAN.md`                      | Updated UI Enablement section + request flow diagram         |

---

## Backward Compatibility

✅ **Fully backward compatible**

- Existing requests without `rlm_enabled` parameter work as before
- Falls back to environment variable default
- No breaking changes to API contract

---

## Next Steps

### Front-End Implementation

The Front-end team needs to:

1. **Add UI Toggle**: Create RLM toggle in Orchestration tab → Advanced/Experimental options
2. **Update Submission**: Pass `rlm_enabled` in `WorkflowRequest` when user toggles RLM
3. **Display Feedback**: Show visual indicator that RLM is active (optional)

### Example UI Code

```typescript
// In Orchestration component
const [rlmEnabled, setRlmEnabled] = useState(false);

const handleSubmit = async (goal: string) => {
  const request: WorkflowRequest = {
    goal,
    use_graph: false,
    rlm_enabled: rlmEnabled  // Pass toggle state
  };

  await executeWorkflow(request);
};

// In Advanced/Experimental options section
<label>
  <input
    type="checkbox"
    checked={rlmEnabled}
    onChange={(e) => setRlmEnabled(e.target.checked)}
  />
  Enable RLM (Recursive Language Model)
</label>
```

---

## Summary

Phase 1 now correctly implements per-execution RLM control:

✅ Global default via `RLM_ENABLED` env var  
✅ Per-execution override via request parameter  
✅ Front-end types updated to support toggle  
✅ Backend routing logic in place  
✅ Backward compatible  
✅ Ready for Orchestration panel integration

**Phase 1 is ready for:** Front-end integration + deployment
