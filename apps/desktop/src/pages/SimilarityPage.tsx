import type { LibrarySelection } from "@anime-pic-manage/shared-types";
import { SimilarityWorkbench } from "../components/SimilarityWorkbench";

export function SimilarityPage({
  library,
  onChooseLibrary,
  libraryLocked = false,
}: {
  library: LibrarySelection | null;
  onChooseLibrary: () => void;
  libraryLocked?: boolean;
}) {
  return (
    <SimilarityWorkbench
      library={library}
      onChooseLibrary={onChooseLibrary}
      libraryLocked={libraryLocked}
    />
  );
}
