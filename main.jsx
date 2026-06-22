import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  AlertTriangle,
  Bolt,
  Boxes,
  CheckCircle2,
  ClipboardCheck,
  Cpu,
  Database,
  Gauge,
  Layers3,
  Lightbulb,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  XCircle,
} from 'lucide-react';
import './styles.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

const FALLBACK_FEATURES = [
  { key: 'wall_thickness_mm', label: 'Wall thickness', unit: 'mm', min: 0.5, max: 4, step: 0.1, default: 2.4, group: 'Geometry' },
  { key: 'fillet_radius_mm', label: 'Fillet radius', unit: 'mm', min: 0.2, max: 2.2, step: 0.05, default: 1, group: 'Geometry' },
  { key: 'draft_angle_deg', label: 'Draft angle', unit: 'deg', min: 0.3, max: 3.2, step: 0.1, default: 2, group: 'Geometry' },
  { key: 'hole_diameter_mm', label: 'Hole diameter', unit: 'mm', min: 2, max: 16, step: 0.1, default: 8, group: 'Geometry' },
  { key: 'rib_height_mm', label: 'Rib height', unit: 'mm', min: 5, max: 24, step: 0.1, default: 9, group: 'Ribs' },
  { key: 'rib_thickness_mm', label: 'Rib thickness', unit: 'mm', min: 1.5, max: 5.2, step: 0.1, default: 3.2, group: 'Ribs' },
  { key: 'tolerance_mm', label: 'Tolerance', unit: 'mm', min: 0.02, max: 0.2, step: 0.005, default: 0.05, group: 'Manufacturing' },
  { key: 'material_density', label: 'Material density', unit: 'g/cc', min: 1, max: 8, step: 0.01, default: 1.18, group: 'Material' },
  { key: 'surface_finish_ra', label: 'Surface finish Ra', unit: 'um', min: 0.6, max: 4.2, step: 0.05, default: 1.2, group: 'Manufacturing' },
  { key: 'assembly_clearance_mm', label: 'Assembly clearance', unit: 'mm', min: 0.04, max: 0.55, step: 0.01, default: 0.3, group: 'Assembly' },
  { key: 'overhang_angle_deg', label: 'Overhang angle', unit: 'deg', min: 28, max: 62, step: 0.5, default: 38, group: 'Manufacturing' },
  { key: 'min_feature_size_mm', label: 'Min feature size', unit: 'mm', min: 0.25, max: 2.5, step: 0.05, default: 1.5, group: 'Manufacturing' },
  { key: 'aspect_ratio', label: 'Part aspect ratio', unit: ':1', min: 2, max: 9.5, step: 0.1, default: 3.2, group: 'Structure' },
  { key: 'part_weight_kg', label: 'Part weight', unit: 'kg', min: 0.15, max: 2.4, step: 0.01, default: 0.45, group: 'Material' },
  { key: 'vent_aspect_ratio', label: 'Vent aspect ratio', unit: ':1', min: 0, max: 7.5, step: 0.1, default: 3, group: 'Product-specific' },
  { key: 'cooling_channel_mm', label: 'Cooling channel', unit: 'mm', min: 0, max: 5.8, step: 0.1, default: 0, group: 'Product-specific' },
  { key: 'ip_groove_mm', label: 'IP groove', unit: 'mm', min: 0, max: 2.1, step: 0.1, default: 0, group: 'Product-specific' },
];

const PRODUCT_ICONS = {
  LIGHTING: Lightbulb,
  EV: Bolt,
  ADAS: Cpu,
  STRUCTURAL: Boxes,
};

const STATUS_ICONS = {
  PASS: CheckCircle2,
  WARNING: AlertTriangle,
  FAIL: XCircle,
};

function defaultsFrom(definitions) {
  return Object.fromEntries(definitions.map((field) => [field.key, field.default ?? 0]));
}

function classForVerdict(verdict) {
  return `verdict ${String(verdict || 'idle').toLowerCase()}`;
}

function groupFields(fields) {
  return fields.reduce((groups, field) => {
    const group = field.group || 'Design';
    groups[group] = groups[group] || [];
    groups[group].push(field);
    return groups;
  }, {});
}

function App() {
  const [metadata, setMetadata] = useState(null);
  const [apiState, setApiState] = useState('checking');
  const [error, setError] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [result, setResult] = useState(null);
  const [form, setForm] = useState({
    design_id: 'CUSTOM-001',
    product_type: 'LIGHTING',
    description: 'Custom CAD design',
    features: defaultsFrom(FALLBACK_FEATURES),
  });

  const featureDefinitions = metadata?.feature_definitions || FALLBACK_FEATURES;
  const groupedFields = useMemo(() => groupFields(featureDefinitions), [featureDefinitions]);
  const productTypes = metadata?.product_types || ['LIGHTING', 'EV', 'ADAS', 'STRUCTURAL'];
  const productLabels = metadata?.product_labels || {};

  useEffect(() => {
    let alive = true;
    fetch(`${API_BASE}/metadata`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (!alive) return;
        setMetadata(data);
        setApiState('online');
        const sample = data.samples?.[2] || data.samples?.[0];
        if (sample) {
          loadSample(sample, false);
        }
      })
      .catch((fetchError) => {
        if (!alive) return;
        setApiState('offline');
        setError(fetchError.message);
      });
    return () => {
      alive = false;
    };
  }, []);

  function updateFeature(key, value) {
    setForm((current) => ({
      ...current,
      features: {
        ...current.features,
        [key]: Number(value),
      },
    }));
  }

  function updateField(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function loadSample(sample, clearResult = true) {
    setForm({
      design_id: sample.design_id,
      product_type: sample.product_type,
      description: sample.description,
      features: { ...defaultsFrom(featureDefinitions), ...sample.features },
    });
    if (clearResult) {
      setResult(null);
    }
  }

  function resetForm() {
    setForm({
      design_id: 'CUSTOM-001',
      product_type: 'LIGHTING',
      description: 'Custom CAD design',
      features: defaultsFrom(featureDefinitions),
    });
    setResult(null);
    setError('');
  }

  async function validateDesign() {
    setIsValidating(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const body = await response.json();
      if (!response.ok) {
        throw new Error(body.detail || `Validation failed with ${response.status}`);
      }
      setResult(body);
      setApiState('online');
    } catch (validationError) {
      setApiState('offline');
      setError(validationError.message);
    } finally {
      setIsValidating(false);
    }
  }

  const fusion = result?.fusion;
  const FusionIcon = STATUS_ICONS[fusion?.verdict] || Activity;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">SmartCAD-AI</span>
          <h1>CAD Design Validation</h1>
        </div>
        <div className={`api-pill ${apiState}`}>
          <span />
          API {apiState}
        </div>
      </header>

      <section className="metrics-row" aria-label="Model summary">
        <Metric icon={Database} label="Training designs" value={metadata?.dataset?.rows || 54} />
        <Metric icon={ShieldCheck} label="Varroc rules" value={metadata?.rules?.length || 12} />
        <Metric icon={Gauge} label="GBM CV mean" value={`${Math.round((metadata?.model_card?.cv_mean || 0.9) * 100)}%`} />
        <Metric icon={Layers3} label="Validation layers" value="3" />
      </section>

      <section className="workspace-grid">
        <div className="panel input-panel">
          <div className="panel-header">
            <div>
              <h2>Design Input</h2>
              <p>{productLabels[form.product_type] || form.product_type}</p>
            </div>
            <button type="button" className="icon-button" onClick={resetForm} title="Reset form" aria-label="Reset form">
              <RotateCcw size={18} />
            </button>
          </div>

          <div className="identity-grid">
            <label>
              <span>Design ID</span>
              <input value={form.design_id} onChange={(event) => updateField('design_id', event.target.value)} />
            </label>
            <label>
              <span>Description</span>
              <input value={form.description} onChange={(event) => updateField('description', event.target.value)} />
            </label>
          </div>

          <div className="segmented" role="tablist" aria-label="Product type">
            {productTypes.map((productType) => {
              const Icon = PRODUCT_ICONS[productType] || ClipboardCheck;
              return (
                <button
                  key={productType}
                  type="button"
                  className={productType === form.product_type ? 'active' : ''}
                  onClick={() => updateField('product_type', productType)}
                >
                  <Icon size={16} />
                  <span>{productType}</span>
                </button>
              );
            })}
          </div>

          {metadata?.samples?.length > 0 && (
            <div className="sample-row">
              {metadata.samples.map((sample) => (
                <button key={sample.design_id} type="button" onClick={() => loadSample(sample)}>
                  <ClipboardCheck size={16} />
                  {sample.design_id}
                </button>
              ))}
            </div>
          )}

          <div className="field-groups">
            {Object.entries(groupedFields).map(([group, fields]) => (
              <fieldset key={group}>
                <legend>{group}</legend>
                <div className="field-grid">
                  {fields.map((field) => (
                    <NumberField
                      key={field.key}
                      field={field}
                      value={form.features[field.key] ?? field.default ?? 0}
                      onChange={updateFeature}
                    />
                  ))}
                </div>
              </fieldset>
            ))}
          </div>

          <button type="button" className="primary-action" onClick={validateDesign} disabled={isValidating}>
            {isValidating ? <RefreshCw className="spin" size={18} /> : <Play size={18} />}
            {isValidating ? 'Validating' : 'Validate Design'}
          </button>
        </div>

        <div className="panel result-panel">
          <div className="panel-header">
            <div>
              <h2>Validation Report</h2>
              <p>{result ? result.design_id : 'Awaiting validation'}</p>
            </div>
            <div className={classForVerdict(fusion?.verdict)}>
              <FusionIcon size={18} />
              {fusion?.verdict || 'READY'}
            </div>
          </div>

          {error && (
            <div className="notice">
              <AlertTriangle size={18} />
              <span>{error}</span>
            </div>
          )}

          {result ? (
            <Report result={result} />
          ) : (
            <div className="empty-state">
              <Activity size={34} />
              <h3>No report yet</h3>
              <p>Select a sample or enter dimensions, then run validation.</p>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

function Metric({ icon: Icon, label, value }) {
  return (
    <div className="metric">
      <Icon size={18} />
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function NumberField({ field, value, onChange }) {
  const fixedValue = Number.isFinite(Number(value)) ? Number(value) : 0;
  return (
    <label className="number-field">
      <span>
        {field.label}
        <em>{field.unit}</em>
      </span>
      <div className="number-controls">
        <input
          type="range"
          min={field.min}
          max={field.max}
          step={field.step}
          value={fixedValue}
          onChange={(event) => onChange(field.key, event.target.value)}
        />
        <input
          type="number"
          min={field.min}
          max={field.max}
          step={field.step}
          value={fixedValue}
          onChange={(event) => onChange(field.key, event.target.value)}
        />
      </div>
    </label>
  );
}

function Report({ result }) {
  const { rule_result: rules, ml_result: ml, fusion } = result;
  const verdicts = [
    { label: 'Rules', verdict: rules.rule_verdict, detail: rules.risk_level },
    { label: 'GBM', verdict: ml.ml_verdict, detail: `${ml.confidence}%` },
    { label: 'Fusion', verdict: fusion.verdict, detail: `${fusion.confidence}%` },
  ];

  return (
    <div className="report">
      <section className="decision-band">
        <div>
          <span>Final verdict</span>
          <strong>{fusion.verdict}</strong>
        </div>
        <div className="confidence-ring" style={{ '--score': `${fusion.confidence}%` }}>
          <span>{fusion.confidence}%</span>
        </div>
      </section>

      <p className="method">{fusion.method}</p>

      <div className="layer-flow">
        {verdicts.map((item) => {
          const Icon = STATUS_ICONS[item.verdict] || Activity;
          return (
            <div key={item.label} className={classForVerdict(item.verdict)}>
              <Icon size={18} />
              <span>{item.label}</span>
              <strong>{item.verdict}</strong>
              <small>{item.detail}</small>
            </div>
          );
        })}
      </div>

      <section className="probabilities">
        <Probability label="Pass probability" value={ml.pass_prob} tone="pass" />
        <Probability label="Fail probability" value={ml.fail_prob} tone="fail" />
      </section>

      <section className="rule-summary">
        <div>
          <span>Critical</span>
          <strong>{rules.critical_count}</strong>
        </div>
        <div>
          <span>Major</span>
          <strong>{rules.major_count}</strong>
        </div>
        <div>
          <span>Minor</span>
          <strong>{rules.minor_count}</strong>
        </div>
        <div>
          <span>Skipped</span>
          <strong>{rules.skipped_rules.length}</strong>
        </div>
      </section>

      <section className="violations">
        <h3>Rule Violations</h3>
        {rules.violations.length ? (
          rules.violations.map((violation) => (
            <article key={violation.rule_id}>
              <div className={`severity ${violation.severity.toLowerCase()}`}>{violation.severity}</div>
              <div>
                <strong>{violation.rule_name}</strong>
                <span>{violation.standard}</span>
                <p>{violation.message}</p>
              </div>
            </article>
          ))
        ) : (
          <p className="quiet">All applicable Varroc rules are satisfied.</p>
        )}
      </section>
    </div>
  );
}

function Probability({ label, value, tone }) {
  return (
    <div className="probability">
      <div>
        <span>{label}</span>
        <strong>{value}%</strong>
      </div>
      <div className="track">
        <span className={tone} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);

