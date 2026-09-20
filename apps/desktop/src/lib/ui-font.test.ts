import { describe, expect, it } from "vitest";
import {
  applyUiFontSize,
  clampUiFontSize,
  stepUiFontSize,
  uiFontStepValue,
  UI_FONT_BASE_PX,
} from "./ui-font";

describe("ui font scaling", () => {
  it("keeps the baseline at 14px", () => {
    expect(clampUiFontSize(UI_FONT_BASE_PX)).toBe(14);
    expect(uiFontStepValue(UI_FONT_BASE_PX)).toBe("1.0000px");
  });

  it("clamps and snaps to 0.5px steps", () => {
    expect(clampUiFontSize(11)).toBe(12);
    expect(clampUiFontSize(99)).toBe(20);
    expect(clampUiFontSize(14.24)).toBe(14);
    expect(clampUiFontSize(14.26)).toBe(14.5);
    expect(clampUiFontSize(Number.NaN)).toBe(14);
  });

  it("steps by 0.5px in both directions", () => {
    expect(stepUiFontSize(14, 1)).toBe(14.5);
    expect(stepUiFontSize(14, -1)).toBe(13.5);
    expect(stepUiFontSize(12, -1)).toBe(12);
    expect(stepUiFontSize(20, 1)).toBe(20);
  });

  it("writes the scale into the document root", () => {
    const values: Record<string, string> = {};
    const root = {
      style: {
        setProperty: (name: string, value: string) => {
          values[name] = value;
        },
      },
    } as unknown as HTMLElement;

    applyUiFontSize(17.5, root);

    expect(values["--ui-font-step"]).toBe("1.2500px");
  });
});
