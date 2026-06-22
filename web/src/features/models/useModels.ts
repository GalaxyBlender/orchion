import { useEffect, useRef, useState } from "react";
import { fetchModels } from "@/shared/api/client";
import { parseNetworkError } from "@/shared/api/errors";
import { ApiRequestError, type ApiSettings, type ModelObject } from "@/shared/api/types";
import { classifyModels, type ClassifiedModels } from "./modelCatalog";

export interface UseModelsState {
  models: ModelObject[];
  classified: ClassifiedModels;
  isLoading: boolean;
  error: ApiRequestError | null;
  reload: () => void;
}

export function useModels(settings: ApiSettings): UseModelsState {
  const [models, setModels] = useState<ModelObject[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<ApiRequestError | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const requestIdRef = useRef(0);

  const reload = () => {
    setReloadToken((token) => token + 1);
  };

  useEffect(() => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    let isCurrent = true;

    setIsLoading(true);
    setError(null);

    fetchModels(settings)
      .then((modelList) => {
        if (!isCurrent || requestId !== requestIdRef.current) {
          return;
        }

        setModels(modelList.data);
        setError(null);
        setIsLoading(false);
      })
      .catch((caughtError: unknown) => {
        if (!isCurrent || requestId !== requestIdRef.current) {
          return;
        }

        setModels([]);
        setError(caughtError instanceof ApiRequestError ? caughtError : parseNetworkError(caughtError));
        setIsLoading(false);
      });

    return () => {
      isCurrent = false;
    };
  }, [settings.apiKey, reloadToken]);

  return { models, classified: classifyModels(models), isLoading, error, reload };
}
