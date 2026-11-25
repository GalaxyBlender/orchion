import { useEffect, useState } from "react";
import { changeModelResidency, fetchModelStatuses } from "@/shared/api/client";
import { parseNetworkError } from "@/shared/api/errors";
import type { ApiSettings, ModelControlAction, ModelControlRequest, ModelStatus } from "@/shared/api/types";

const STATUS_REFRESH_MS = 5_000;

type ModelOperation = ModelControlRequest & { action: ModelControlAction };

export function useModelLifecycle(settings: ApiSettings, onChanged: () => void) {
  const [statuses, setStatuses] = useState<ModelStatus[]>([]);
  const [isRefreshing, setIsRefreshing] = useState(true);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [operationError, setOperationError] = useState<(ModelOperation & { message: string }) | null>(null);
  const [pending, setPending] = useState<ModelOperation | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    // Pause polling while a command is pending. After it settles, read the whole list:
    // loading one model can evict a different one.
    if (pending) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    const refresh = async () => {
      try {
        const { data } = await fetchModelStatuses(settings, controller.signal);
        if (controller.signal.aborted) return;
        setStatuses(data);
        setStatusError(null);
      } catch (error) {
        if (controller.signal.aborted) return;
        setStatuses([]);
        setStatusError(parseNetworkError(error).message);
      } finally {
        if (!controller.signal.aborted) {
          setIsRefreshing(false);
          timer = setTimeout(refresh, STATUS_REFRESH_MS);
        }
      }
    };

    void refresh();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [settings, reloadToken, pending]);

  function reload() {
    setIsRefreshing(true);
    setReloadToken((token) => token + 1);
  }

  async function change(action: ModelControlAction, target: ModelControlRequest) {
    if (pending || isRefreshing) return;
    const operation = { ...target, action };
    setPending(operation);
    setOperationError(null);

    try {
      await changeModelResidency(settings, action, target);
    } catch (error) {
      setOperationError({ ...operation, message: parseNetworkError(error).message });
      return;
    } finally {
      setIsRefreshing(true);
      setPending(null);
      setReloadToken((token) => token + 1);
    }
    onChanged();
  }

  return { statuses, isRefreshing, statusError, pending, operationError, reload, change };
}

export type ModelLifecycle = ReturnType<typeof useModelLifecycle>;
