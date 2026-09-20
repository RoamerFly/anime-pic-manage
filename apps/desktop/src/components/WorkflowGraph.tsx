import { useMemo, useState } from "react";
import { RotateCcw, ZoomIn, ZoomOut } from "lucide-react";

export type NodeOverrides = Record<string, Record<string, unknown>>;

export interface WorkflowRow {
  name: string;
  /** Link reference `[nodeId, outputIndex]` or a literal value. */
  link: { nodeId: string; outputIndex: number } | null;
  value: unknown;
  height: number;
  /** Vertical centre of this row inside the node card. */
  offsetY: number;
}

export interface WorkflowNode {
  id: string;
  classType: string;
  inputs: Record<string, unknown>;
  rows: WorkflowRow[];
  depth: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface WorkflowEdge {
  from: string;
  to: string;
  input: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface WorkflowLayout {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  width: number;
  height: number;
}

export const NODE_WIDTH = 260;
const HEADER_HEIGHT = 40;
const ROW_PADDING = 8;
const COLUMN_GAP = 120;
const ROW_GAP = 16;
const LINK_ROW_HEIGHT = 20;
const SCALAR_ROW_HEIGHT = 32;
const TEXTAREA_EXTRA = 40;

/** Accept both our envelope and a raw API-format graph. */
export function graphNodesOf(template: unknown): Record<string, Record<string, unknown>> {
  const source = (template ?? {}) as Record<string, unknown>;
  const candidate =
    source.prompt && typeof source.prompt === "object"
      ? (source.prompt as Record<string, unknown>)
      : source;
  const nodes: Record<string, Record<string, unknown>> = {};
  for (const [id, value] of Object.entries(candidate)) {
    if (value && typeof value === "object" && "class_type" in (value as object)) {
      nodes[id] = value as Record<string, unknown>;
    }
  }
  return nodes;
}

/**
 * ComfyUI writes link references as `[nodeId, outputIndex]`. The id is usually
 * a string but some workflows (and hand-written graphs) use a number, so both
 * have to be accepted or the edge silently disappears.
 */
export function linkOf(value: unknown): { nodeId: string; outputIndex: number } | null {
  if (!Array.isArray(value) || value.length < 1) return null;
  const [id, output] = value;
  if (typeof id !== "string" && typeof id !== "number") return null;
  return {
    nodeId: String(id),
    outputIndex: typeof output === "number" ? output : 0,
  };
}

function rowHeight(value: unknown): number {
  if (linkOf(value) !== null) return LINK_ROW_HEIGHT;
  if (typeof value === "string" && (value.length > 48 || value.includes("\n"))) {
    return SCALAR_ROW_HEIGHT + TEXTAREA_EXTRA;
  }
  return SCALAR_ROW_HEIGHT;
}

function buildRows(inputs: Record<string, unknown>): WorkflowRow[] {
  let offset = HEADER_HEIGHT;
  return Object.entries(inputs).map(([name, value]) => {
    const height = rowHeight(value);
    const offsetY = offset + height / 2;
    offset += height;
    return { name, link: linkOf(value), value, height, offsetY };
  });
}

/** Longest-path layering, then per-column stacking using real card heights. */
export function layoutWorkflow(template: unknown): WorkflowLayout {
  const raw = graphNodesOf(template);
  const depths = new Map<string, number>();
  const visit = (id: string, seen: Set<string>): number => {
    const cached = depths.get(id);
    if (cached !== undefined) return cached;
    if (seen.has(id)) return 0; // defensive: the graph is acyclic
    const node = raw[id];
    if (!node) return 0;
    seen.add(id);
    const parents = buildRows((node.inputs ?? {}) as Record<string, unknown>)
      .map((row) => row.link?.nodeId)
      .filter((parent): parent is string => parent !== undefined && parent in raw);
    const depth = parents.length === 0 ? 0 : Math.max(...parents.map((p) => visit(p, seen))) + 1;
    seen.delete(id);
    depths.set(id, depth);
    return depth;
  };
  for (const id of Object.keys(raw)) visit(id, new Set());

  const columns = new Map<number, number>();
  const nodes: WorkflowNode[] = Object.entries(raw).map(([id, node]) => {
    const rows = buildRows((node.inputs ?? {}) as Record<string, unknown>);
    const height = HEADER_HEIGHT + rows.reduce((total, row) => total + row.height, 0) + ROW_PADDING;
    const depth = depths.get(id) ?? 0;
    const y = columns.get(depth) ?? 0;
    columns.set(depth, y + height + ROW_GAP);
    return {
      id,
      classType: String(node.class_type ?? "?"),
      inputs: (node.inputs ?? {}) as Record<string, unknown>,
      rows,
      depth,
      x: depth * (NODE_WIDTH + COLUMN_GAP),
      y,
      width: NODE_WIDTH,
      height,
    };
  });
  nodes.sort((left, right) => left.depth - right.depth || left.y - right.y);

  const byId = new Map(nodes.map((node) => [node.id, node]));
  const edges: WorkflowEdge[] = [];
  for (const node of nodes) {
    for (const row of node.rows) {
      const source = row.link ? byId.get(row.link.nodeId) : undefined;
      if (!row.link || !source) continue;
      edges.push({
        from: source.id,
        to: node.id,
        input: row.name,
        x1: source.x + source.width,
        y1: source.y + HEADER_HEIGHT / 2,
        x2: node.x,
        y2: node.y + row.offsetY,
      });
    }
  }

  const width = nodes.reduce((max, node) => Math.max(max, node.x + node.width), NODE_WIDTH) + 40;
  const height = nodes.reduce((max, node) => Math.max(max, node.y + node.height), 120) + 40;
  return { nodes, edges, width, height };
}

export function WorkflowGraph({
  template,
  overrides,
  onChange,
}: {
  template: unknown;
  overrides: NodeOverrides;
  onChange: (nodeId: string, input: string, value: unknown) => void;
}) {
  const [zoom, setZoom] = useState(1);
  const [hovered, setHovered] = useState<string | null>(null);
  const layout = useMemo(() => layoutWorkflow(template), [template]);

  if (layout.nodes.length === 0) {
    return <small>这个模板没有可显示的节点。</small>;
  }

  return (
    <div className="workflow-graph">
      <div className="workflow-graph-toolbar">
        <span className="eyebrow">
          {layout.nodes.length} 个节点 · {layout.edges.length} 条连线
        </span>
        <button
          type="button"
          className="ghost-button compact"
          onClick={() => setZoom((value) => Math.max(0.4, value - 0.1))}
          title="缩小"
        >
          <ZoomOut size={14} />
        </button>
        <span className="workflow-graph-zoom">{Math.round(zoom * 100)}%</span>
        <button
          type="button"
          className="ghost-button compact"
          onClick={() => setZoom((value) => Math.min(1.6, value + 0.1))}
          title="放大"
        >
          <ZoomIn size={14} />
        </button>
        <small>
          连线按“节点输出 → 具体输入行”绘制；改动只保存在本机，蓝色标记表示已覆盖。
        </small>
      </div>
      <div className="workflow-graph-canvas">
        <div
          className="workflow-graph-stage"
          style={{ width: layout.width * zoom, height: layout.height * zoom }}
        >
          <div
            style={{
              width: layout.width,
              height: layout.height,
              transform: `scale(${zoom})`,
              transformOrigin: "top left",
              position: "relative",
            }}
          >
            <svg
              className="workflow-graph-edges"
              width={layout.width}
              height={layout.height}
            >
              <defs>
                <marker
                  id="workflow-arrow"
                  viewBox="0 0 8 8"
                  refX="7"
                  refY="4"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 8 4 L 0 8 z" fill="#8fa2d8" />
                </marker>
              </defs>
              {layout.edges.map((edge) => {
                const active = hovered === edge.to || hovered === edge.from;
                const mid = (edge.x1 + edge.x2) / 2;
                return (
                  <path
                    key={`${edge.from}-${edge.to}-${edge.input}`}
                    d={`M ${edge.x1} ${edge.y1} C ${mid} ${edge.y1}, ${mid} ${edge.y2}, ${edge.x2} ${edge.y2}`}
                    fill="none"
                    stroke={active ? "#4d68d8" : "#8fa2d8"}
                    strokeWidth={active ? 2.2 : 1.5}
                    markerEnd="url(#workflow-arrow)"
                  />
                );
              })}
            </svg>
            {layout.nodes.map((node) => {
              const overridden = Object.keys(overrides[node.id] ?? {}).length > 0;
              return (
                <div
                  key={node.id}
                  className={`workflow-node${overridden ? " edited" : ""}`}
                  style={{
                    left: node.x,
                    top: node.y,
                    width: node.width,
                    height: node.height,
                  }}
                  onMouseEnter={() => setHovered(node.id)}
                  onMouseLeave={() => setHovered((current) => (current === node.id ? null : current))}
                >
                  <div className="workflow-node-header">
                    <strong>{node.classType}</strong>
                    <code>#{node.id}</code>
                  </div>
                  {node.rows.map((row) => {
                    if (row.link) {
                      return (
                        <div
                          className="workflow-node-link"
                          key={row.name}
                          style={{ minHeight: row.height }}
                        >
                          <span>{row.name}</span>
                          <code>← #{row.link.nodeId}</code>
                        </div>
                      );
                    }
                    const edited =
                      overrides[node.id] !== undefined && row.name in (overrides[node.id] ?? {});
                    const long =
                      typeof row.value === "string" &&
                      (row.value.length > 48 || row.value.includes("\n"));
                    return (
                      <label
                        className={`workflow-node-input${edited ? " edited" : ""}`}
                        key={row.name}
                        style={{ minHeight: row.height }}
                      >
                        <span>
                          {row.name}
                          {edited && (
                            <button
                              type="button"
                              className="workflow-node-reset"
                              title="还原为模板值"
                              onClick={() => onChange(node.id, row.name, undefined)}
                            >
                              <RotateCcw size={11} />
                            </button>
                          )}
                        </span>
                        {typeof row.value === "number" ? (
                          <input
                            type="number"
                            value={String(row.value)}
                            onChange={(event) =>
                              onChange(
                                node.id,
                                row.name,
                                event.target.value === "" ? 0 : Number(event.target.value),
                              )
                            }
                          />
                        ) : typeof row.value === "boolean" ? (
                          <input
                            type="checkbox"
                            checked={row.value}
                            onChange={(event) =>
                              onChange(node.id, row.name, event.target.checked)
                            }
                          />
                        ) : typeof row.value === "string" ? (
                          long ? (
                            <textarea
                              rows={3}
                              value={row.value}
                              onChange={(event) =>
                                onChange(node.id, row.name, event.target.value)
                              }
                            />
                          ) : (
                            <input
                              type="text"
                              value={row.value}
                              onChange={(event) =>
                                onChange(node.id, row.name, event.target.value)
                              }
                            />
                          )
                        ) : (
                          <code>{JSON.stringify(row.value)}</code>
                        )}
                      </label>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
