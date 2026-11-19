import { useCallback, useMemo, useState } from 'react';

export interface CrackSelectionModifiers {
  shiftKey?: boolean;
  ctrlKey?: boolean;
  metaKey?: boolean;
}

/** Shared selection state for every crack review surface (canvas and findings list). */
export function useCrackSelection() {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const [anchorId, setAnchorId] = useState<string | null>(null);
  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);

  const clear = useCallback(() => {
    setSelectedIds([]);
    setFocusedId(null);
    setAnchorId(null);
  }, []);

  const selectSingle = useCallback((id: string | null) => {
    setSelectedIds(id ? [id] : []);
    setFocusedId(id);
    setAnchorId(id);
  }, []);

  const selectMany = useCallback((ids: string[]) => {
    const unique = [...new Set(ids)];
    setSelectedIds(unique);
    setFocusedId((previous) => previous && unique.includes(previous) ? previous : unique.length === 1 ? unique[0] : null);
    setAnchorId(unique[0] ?? null);
  }, []);

  const selectGesture = useCallback((id: string, orderedIds: readonly string[], modifiers: CrackSelectionModifiers = {}) => {
    const { shiftKey = false, ctrlKey = false, metaKey = false } = modifiers;
    const additive = ctrlKey || metaKey;
    if (shiftKey) {
      const anchor = anchorId && orderedIds.includes(anchorId) ? anchorId : id;
      const a = orderedIds.indexOf(anchor);
      const b = orderedIds.indexOf(id);
      if (a < 0 || b < 0) {
        setSelectedIds([id]);
      } else {
        const range = orderedIds.slice(Math.min(a, b), Math.max(a, b) + 1);
        setSelectedIds(range);
      }
      setFocusedId(id);
      if (!anchorId) setAnchorId(anchor);
      return;
    }
    if (additive) {
      setSelectedIds((previous) => previous.includes(id) ? previous.filter((item) => item !== id) : [...previous, id]);
      setFocusedId((previous) => previous === id ? null : id);
      setAnchorId(id);
      return;
    }
    setSelectedIds([id]);
    setFocusedId(id);
    setAnchorId(id);
  }, [anchorId]);

  const selectAll = useCallback((ids: readonly string[]) => {
    const unique = [...new Set(ids)];
    setSelectedIds(unique);
    setFocusedId((previous) => previous && unique.includes(previous) ? previous : null);
    setAnchorId(unique[0] ?? null);
  }, []);

  const invert = useCallback((scopeIds: readonly string[]) => {
    setSelectedIds((previous) => {
      const selected = new Set(previous);
      return scopeIds.filter((id) => !selected.has(id));
    });
    setFocusedId(null);
    setAnchorId(scopeIds[0] ?? null);
  }, []);

  const remove = useCallback((ids: readonly string[]) => {
    const removed = new Set(ids);
    setSelectedIds((previous) => previous.filter((id) => !removed.has(id)));
    setFocusedId((previous) => previous && !removed.has(previous) ? previous : null);
    setAnchorId((previous) => previous && !removed.has(previous) ? previous : null);
  }, []);

  return { selectedIds, selectedSet, focusedId, selectionCount: selectedIds.length, clear, selectSingle, selectMany, selectGesture, selectAll, invert, remove };
}
