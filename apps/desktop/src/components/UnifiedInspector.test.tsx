import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { UnifiedInspector, type InspectorPerson } from "./UnifiedInspector";

const people: InspectorPerson[] = [
  {
    id: "person-1",
    title: "artoria_pendragon_(fate)",
    confidence: 0.06,
    status: "unknown",
    margin: 0.05,
    consistency: 1,
    personIndex: 0,
    candidates: [
      { label: "artoria_pendragon_(fate)", score: 0.06 },
      { label: "saber", score: 0.04 },
    ],
  },
];

function render(assignedLabel: string) {
  return renderToStaticMarkup(
    <UnifiedInspector
      people={people}
      activePersonId="person-1"
      onSelectPerson={() => undefined}
      filter="all"
      onChangeFilter={() => undefined}
      editorMode
      onToggleEditor={() => undefined}
      onAddNewBox={() => undefined}
      activeAnnotation={{
        clientId: "client-1",
        label_name: assignedLabel,
        bbox: { x: 0, y: 0, width: 1, height: 1 },
        source: "manual",
        manually_adjusted: false,
      }}
      labelDraft={assignedLabel}
      onChangeLabelDraft={() => undefined}
      onCommitLabel={() => undefined}
      onRemoveAnnotation={() => undefined}
      identities={[]}
      canUndo={false}
      canRedo={false}
      dirty={false}
      onUndo={() => undefined}
      onRedo={() => undefined}
      onCancelEdits={() => undefined}
      onSaveAnnotations={() => undefined}
      saveLoading={false}
    />,
  );
}

describe("UnifiedInspector candidate list", () => {
  it("no longer renders the circular confidence card", () => {
    const markup = render("");

    expect(markup).not.toContain("confidence-ring");
    expect(markup).not.toContain("inspector-person-hero");
    expect(markup).not.toContain("Margin");
  });

  it("highlights the assigned candidate instead", () => {
    const markup = render("saber");

    expect(markup).toContain("inspector-candidate-row assigned");
    expect(markup).toContain("当前已指派为: saber");
    // Only the assigned row carries the highlight class.
    expect(markup.match(/inspector-candidate-row assigned/g)).toHaveLength(1);
  });

  it("highlights nothing while the annotation has no label", () => {
    expect(render("")).not.toContain("inspector-candidate-row assigned");
  });
});
