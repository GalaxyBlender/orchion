use crate::OcrAssets;
use crate::result::{build_vl_runtime, run_vl_ocr};
use orchion_core::{
    DevicePreference, KnownOcrModel, OcrLimits, OcrOptions, OcrResult, OrchionError, Result,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use tokio::sync::{mpsc, oneshot};

const REQUEST_QUEUE_CAPACITY: usize = 1;

#[derive(Clone)]
pub(crate) struct OcrVlWorker {
    // Drop the last sender before joining so the owner thread can leave its receive loop.
    sender: mpsc::Sender<OcrVlRequest>,
    owner: Arc<OwnerThread>,
    has_layout_predictor: bool,
}

struct OwnerThread {
    join: Mutex<Option<JoinHandle<()>>>,
}

struct OcrVlRequest {
    model: KnownOcrModel,
    image_path: PathBuf,
    options: OcrOptions,
    limits: OcrLimits,
    response: oneshot::Sender<Result<OcrResult>>,
    _owner: Arc<OwnerThread>,
}

impl OcrVlWorker {
    pub(crate) fn load(
        model: KnownOcrModel,
        assets: OcrAssets,
        device: DevicePreference,
    ) -> Result<Self> {
        let has_layout_predictor = matches!(
            &assets,
            OcrAssets::VisionLanguage {
                layout: Some(_),
                ..
            }
        );
        let (sender, mut receiver) = mpsc::channel::<OcrVlRequest>(REQUEST_QUEUE_CAPACITY);
        let (loaded_sender, loaded_receiver) = std::sync::mpsc::sync_channel(1);
        let join = thread::Builder::new()
            .name("orchion-ocr-vl".to_string())
            .spawn(move || {
                let runtime = match build_vl_runtime(model, &assets, device) {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let _ = loaded_sender.send(Err(error));
                        return;
                    }
                };
                if loaded_sender.send(Ok(())).is_err() {
                    return;
                }

                while let Some(request) = receiver.blocking_recv() {
                    let result = run_vl_ocr(
                        request.model,
                        &runtime,
                        &request.image_path,
                        &request.options,
                        request.limits,
                    );
                    let _ = request.response.send(result);
                }
            })
            .map_err(|error| OrchionError::ModelLoad {
                message: format!("failed to start OCR-VL worker thread: {error}"),
            })?;
        let worker = Self {
            sender,
            owner: Arc::new(OwnerThread {
                join: Mutex::new(Some(join)),
            }),
            has_layout_predictor,
        };

        loaded_receiver
            .recv()
            .map_err(|error| OrchionError::ModelLoad {
                message: format!("OCR-VL worker stopped during model loading: {error}"),
            })??;
        Ok(worker)
    }

    pub(crate) const fn has_layout_predictor(&self) -> bool {
        self.has_layout_predictor
    }

    pub(crate) async fn run(
        &self,
        model: KnownOcrModel,
        image_path: PathBuf,
        options: OcrOptions,
        limits: OcrLimits,
    ) -> Result<OcrResult> {
        let (response, result) = oneshot::channel();
        self.sender
            .send(OcrVlRequest {
                model,
                image_path,
                options,
                limits,
                response,
                _owner: Arc::clone(&self.owner),
            })
            .await
            .map_err(|_| OrchionError::Inference {
                message: "OCR-VL worker stopped before accepting the request".to_string(),
            })?;
        result.await.map_err(|_| OrchionError::Inference {
            message: "OCR-VL worker stopped before returning a result".to_string(),
        })?
    }
}

impl Drop for OwnerThread {
    fn drop(&mut self) {
        let join = self
            .join
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(join) = join
            && join.thread().id() != thread::current().id()
        {
            let _ = join.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn cancelled_request_keeps_owner_alive_until_worker_exits() {
        let (sender, mut receiver) = mpsc::channel::<OcrVlRequest>(REQUEST_QUEUE_CAPACITY);
        let (request_received, received) = std::sync::mpsc::sync_channel(1);
        let (release_request, released) = std::sync::mpsc::sync_channel(1);
        let (stopped, worker_stopped) = std::sync::mpsc::sync_channel(1);
        let join = thread::spawn(move || {
            let request = receiver.blocking_recv().unwrap();
            request_received.send(()).unwrap();
            released.recv().unwrap();
            drop(request);
            assert!(receiver.blocking_recv().is_none());
            stopped.send(()).unwrap();
        });
        let worker = OcrVlWorker {
            sender,
            owner: Arc::new(OwnerThread {
                join: Mutex::new(Some(join)),
            }),
            has_layout_predictor: false,
        };
        let (response, result) = oneshot::channel();
        worker
            .sender
            .blocking_send(OcrVlRequest {
                model: KnownOcrModel::PaddleOcrVl16,
                image_path: PathBuf::new(),
                options: OcrOptions::default(),
                limits: OcrLimits::default(),
                response,
                _owner: Arc::clone(&worker.owner),
            })
            .unwrap();

        received.recv().unwrap();
        drop(result);
        let (handle_dropped, dropped) = std::sync::mpsc::sync_channel(1);
        let dropper = thread::spawn(move || {
            drop(worker);
            handle_dropped.send(()).unwrap();
        });
        let dropped_without_waiting = dropped.recv_timeout(Duration::from_secs(1)).is_ok();

        release_request.send(()).unwrap();

        worker_stopped
            .recv_timeout(Duration::from_secs(1))
            .expect("owner thread should stop after a cancelled request");
        dropper.join().unwrap();
        assert!(
            dropped_without_waiting,
            "dropping the last external handle must not wait for a cancelled request"
        );
    }
}
