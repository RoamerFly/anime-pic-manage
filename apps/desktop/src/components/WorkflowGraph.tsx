import { useMemo, useState } from "react";
import { RotateCcw, ZoomIn, ZoomOut } from "lucide-react";

export type NodeOverrides = Record<string, Record<string, unknown>>;

interface GraphNode {
  id: string;
  classType: string;
  inputs: Record<string, unknown>;
  depth: number;
  row: number;
}

const NODE_WIDTH = 236;
const COLUMN_GAP = 96;
const ROW_GAP = 18;

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

function upstreamIds(inputs: Record<string, unknown>): string[] {
  const ids: string[] = [];
  for (const value of Object.values(inputs)) {
    if (Array.isArray(value) && value.length >= 1 && typeof value[0] === "string") {
      ids.push(value[0]);
    }
  }
  return ids;
}

/** Longest-path layering: every node sits right of the nodes it consumes. */
function layout(nodes: Record<string, Record<string, unknown>>): GraphNode[] {
  const depths = new Map<string, number>();
  const visit = (id: string, seen: Set<string>): number => {
    const cached = depths.get(id);
    if (cached !== undefined) return cached;
    if (seen.has(id)) return 0; // defensive: ComfyUI graphs are acyclic
    const node = nodes[id];
    if (!node) return 0;
    seen.add(id);
    const inputs = (node.inputs ?? {}) as Record<string, unknown>;
    const parents = upstreamIds(inputs).filter((parent) => parent in nodes);
    const depth = parents.length === 0 ? 0 : Math.max(...parents.map((p) => visit(p, seen))) + 1;
    seen.delete(id);
    depths.set(id, depth);
    return depth;
  };
  for (const id of Object.keys(nodes)) visit(id, new Set());

  const rows = new Map<number, number>();
  return Object.entries(nodes)
    .map(([id, node]) => {
      const depth = depths.get(id) ?? 0;
      const row = rows.get(depth) ?? 0;
      rows.set(depth, row + 1);
      return {
        id,
        classType: String(node.class_type ?? "?"),
        inputs: (node.inputs ?? {}) as Record<string, unknown>,
        depth,
        row,
      };
    })
    .sort((left, right) => left.depth - right.depth || left.row - right.row);
}

function nodeHeight(node: GraphNode): number {
  const entries = Object.entries(node.inputs);
  const editable = entries.filter(([, value]) => !Array.isArray(value));
  const readOnly = entries.length - editable.length;
  const multiline = editable.filter(
    ([, value]) => typeof value === "string" && (value.length > 48 || value.includes("\n")),
  ).length;
  // 34px header + one line per scalar + extra rows for textareas + footer.
  return 40 + readOnly * 18 + editable.length * 30 + multiline * 34 + 12;
}

function position(node: GraphNode): { x: number; y: number } {
  return {
    x: node.depth * (NODE_WIDTH + COLUMN_GAP),
    y: node.row * (180 + ROW_GAP),
  };
}

function isEdited(overrides: NodeOverrides, nodeId: string, input: string): boolean {
  return overrides[nodeId] !== undefined && input in (overrides[nodeId] ?? {});
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
  const nodes = useMemo(() => layout(graphNodesOf(template)), [template]);
  const width = useMemo(
    () =>
      nodes.reduce((max, node) => Math.max(max, position(node).x + NODE_WIDTH), NODE_WIDTH) + 40,
    [nodes],
  );
  const height = useMemo(() => {
    const bottom = nodes.reduce(
      (max, node) => Math.max(max, position(node).y + nodeHeight(node)),
      180,
    );
    return bottom + 40;
  }, [nodes]);
  const byId = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes]);

  if (nodes.length === 0) {
    return <small>这个模板没有可显示的节点。</small>;
  }

  return (
    <div className="workflow-graph">
      <div className="workflow-graph-toolbar">
        <span className="eyebrow">{nodes.length} 个节点</span>
        <button
          type="button"
          className="ghost-button compact"
          onClick={() => setZoom((value) => Math.max(0.5, value - 0.1))}
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
        <small>改动的值只保存在本机，模板文件不会被修改；蓝色标记表示已覆盖。</small>
      </div>
      <div className="workflow-graph-canvas">
        <div
          className="workflow-graph-stage"
          style={{
            width: width * zoom,
            height: height * zoom,
          }}
        >
          <div
            style={{
              width,
              height,
              transform: `scale(${zoom})`,
              transformOrigin: "top left",
              position: "relative",
            }}
          >
            <svg className="workflow-graph-edges" width={width} height={height}>
              {nodes.flatMap((node) =>
                upstreamIds(node.inputs)
                  .filter((parent) => byId.has(parent))
                  .map((parent, index) => {
                    const from = byId.get(parent)!;
                    const start = position(from);
                    const end = position(node);
                    const x1 = start.x + NODE_WIDTH;
                    const y1 = start.y + 24;
                    const x2 = end.x;
                    const y2 = end.y + 24;
                    const mid = (x1 + x2) / 2;
                    return (
                      <path
                        key={`${parent}-${node.id}-${index}`}
                        d={`M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`}
                        fill="none"
                        stroke="#b9c2d8"
                        strokeWidth={1.4}
                      />
                    );
                  }),
              )}
            </svg>
            {nodes.map((node) => {
              const { x, y } = position(node);
              const overridden = Object.keys(overrides[node.id] ?? {}).length > 0;
              return (
                <div
                  key={node.id}
                  className={`workflow-node${overridden ? " edited" : ""}`}
                  style={{ left: x, top: y, width: NODE_WIDTH }}
                >
                  <div className="workflow-node-header">
                    <strong>{node.classType}</strong>
                    <code>#{node.id}</code>
                  </div>
                  {Object.entries(node.inputs).map(([input, value]) => {
                    if (Array.isArray(value)) {
                      return (
                        <div className="workflow-node-link" key={input}>
                          <span>{input}</span>
                          <code>← #{String(value[0])}</code>
                        </div>
                      );
                    }
                    const edited = isEdited(overrides, node.id, input);
                    const long =
                      typeof value === "string" && (value.length > 48 || value.includes("\n"));
                    return (
                      <label
                        className={`workflow-node-input${edited ? " edited" : ""}`}
                        key={input}
                      >
                        <span>
                          {input}
                          {edited && (
                            <button
                              type="button"
                              className="workflow-node-reset"
                              title="还原为模板值"
                              onClick={() =>
                                onChange(node.id, input, undefined)
                              }
                            >
                              <RotateCcw size={11} />
                            </button>
                          )}
                        </span>
                        {typeof value === "number" ? (
                          <input
                            type="number"
                            value={String(value)}
                            onChange={(event) =>
                              onChange(
                                node.id,
                                input,
                                event.target.value === "" ? 0 : Number(event.target.value),
                              )
                            }
                          />
                        ) : typeof value === "boolean" ? (
                          <input
                            type="checkbox"
                            checked={value}
                            onChange={(event) => onChange(node.id, input, event.target.checked)}
                          />
                        ) : typeof value === "string" ? (
                          long ? (
                            <textarea
                              rows={3}
                              value={value}
                              onChange={(event) => onChange(node.id, input, event.target.value)}
                            />
                          ) : (
                            <input
                              type="text"
                              value={value}
                              onChange={(event) => onChange(node.id, input, event.target.value)}
                            />
                          )
                        ) : (
                          <code>{JSON.stringify(value)}</code>
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
