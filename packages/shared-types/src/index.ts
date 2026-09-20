/** Version of the JSON contract shared by the UI, Rust core, and Python worker. */
export const IPC_SCHEMA_VERSION = "1.0" as const;

export type MessageType =
  | "system.health"
  | "system.gpu.list"
  | "runtime.compute.check"
  | "worker.health"
  | "runtime.capabilities"
  | "library.select"
  | "library.preview.file"
  | "library.scan.start"
  | "library.scan.pause"
  | "library.scan.resume"
  | "library.scan.cancel"
  | "recognition.get_results"
  | "recognition.confirm"
  | "annotation.list"
  | "annotation.save"
  | "training.status"
  | "training.start"
  | "personal.training.status"
  | "personal.training.start"
  | "personal.training.activate"
  | "personal.training.rollback"
  | "personal.training.delete"
  | "personal_model.activate"
  | "personal_model.rollback"
  | "personal_model.delete"
  | "library.files.move"
  | "identity.list"
  | "classification.plan.create"
  | "classification.plan.validate"
  | "classification.plan.execute"
  | "classification.batch.undo"
  | "similarity.scan.start"
  | "similarity.scan.cancel"
  | "similarity.scan.state"
  | "similarity.group.list"
  | "similarity.decision.apply"
  | "comfy.status"
  | "comfy.start"
  | "comfy.stop"
  | "comfy.models"
  | "comfy.templates"
  | "comfy.generate"
  | "comfy.cancel"
  | "comfy.open_output"
  | "dataset.candidates"
  | "dataset.export"
  | "kohya.status"
  | "kohya.probe"
  | "kohya.train"
  | "kohya.status.run"
  | "kohya.stop"
  | "kohya.install"
  | "system.open_url"
  | "cuda.status"
  | "cuda.install"
  | "model.inventory"
  | "model.activate"
  | "model.install"
  | "model.delete"
  | "reference.status"
  | "reference.build"
  | "reference.clear"
  | "model.list"
  | "model.install"
  | "model.activate"
  | "model.health"
  | "settings.get"
  | "settings.update"
  | "job.list"
  | "job.get"
  | "updater.check"
  | "updater.install"
  | "system.explorer.show"
  | "library.file.delete";

export type TaskStatus =
  | "queued"
  | "running"
  | "paused"
  | "completed"
  | "cancelled"
  | "failed"
  | "retrying";

export type ImageStatus =
  | "unscanned"
  | "scanning"
  | "single_character"
  | "multi_character"
  | "needs_review"
  | "unrecognized"
  | "no_character"
  | "failed"
  | "planned"
  | "applied";

export type WorkerStatus = "healthy" | "starting" | "stopped" | "unavailable" | "error";

export type WorkerRuntimeMode = "executable" | "embedded_env";

export type WorkerComputeDevice = "auto" | "cpu" | "cuda";

export type SettingsTab = "environment" | "recognition" | "similarity" | "comfy";

export interface AppSettings {
  /** ONNX Runtime intra-op threads used by CPU recognition inference. */
  recognition_onnx_threads: number;
  /** Default for the recognition scan "skip annotated images" switch. */
  recognition_skip_annotated: boolean;
  /** Threads used to decode and hash images during a similarity scan. */
  similarity_workers: number;
  /** Default match sensitivity for new similarity scans. */
  similarity_threshold: number;
  /** Default for the similarity "include subfolders" switch. */
  similarity_include_subfolders: boolean;
  /** Default archive subfolder name used by the similarity governance bar. */
  similarity_archive_dir: string;
  /** UI font size in pixels against a 14px design baseline. */
  ui_font_size: number;
  /** Root of the user's own ComfyUI portable package; never bundled. */
  comfy_root: string;
  comfy_port: number;
  comfy_low_vram: boolean;
  comfy_auto_start: boolean;
  /** Empty means `<comfy_root>/../Images/generated`. */
  comfy_output_dir: string;
  /** Root of the user's own kohya_ss / sd-scripts install; never bundled. */
  lora_trainer_root: string;
  /** Optional python.exe override for the trainer environment. */
  lora_trainer_python: string;
  /** Base checkpoint used for LoRA training; must match the LoRA family. */
  lora_trainer_base_model: string;
  /** Empty means a `lora-models` folder next to the exported datasets. */
  lora_trainer_output_dir: string;
  /** Optional folder with CUDA 12 / cuDNN 9 DLLs for GPU inference. */
  cuda_runtime_dir: string;
  /** Character recognizer in use; empty means the installed default. */
  recognition_recognizer_model: string;
  /** `below_normal` keeps background jobs polite; `normal` runs flat out. */
  background_priority: "below_normal" | "normal";
  /** Whether scans consult the reference-image library. */
  reference_matching_enabled: boolean;
  /** Similarity backend of that library: CCIP or the recognizer embedding. */
  reference_backend: "ccip" | "embedding";
}

export interface ComfyStatus {
  configured: boolean;
  installed: boolean;
  running: boolean;
  owned: boolean;
  root?: string | null;
  python?: string | null;
  version?: string | null;
  port: number;
  output_dir?: string | null;
  device?: string | null;
  vram_gb?: number | null;
  error?: string | null;
}

export interface ComfyModelList {
  checkpoints: string[];
  loras: string[];
  samplers: string[];
  schedulers: string[];
}

export interface ComfyLoraSelection {
  name: string;
  strength: number;
}

export interface ComfyGenerateRequest {
  template: string;
  checkpoint: string;
  lora?: ComfyLoraSelection | null;
  positive: string;
  negative?: string;
  steps?: number;
  cfg?: number;
  sampler?: string;
  scheduler?: string;
  width?: number;
  height?: number;
  batch?: number;
  seed?: number;
  filename_prefix?: string | null;
}

export interface ComfyGeneratedImage {
  filename: string;
  subfolder: string;
  path: string;
}

export interface ComfyGenerateResult {
  prompt_id: string;
  images: ComfyGeneratedImage[];
  output_dir: string;
  elapsed_ms: number;
  seed: number;
}

export interface ComfyProgress {
  phase: string;
  current: number;
  total: number;
  message: string;
  prompt_id?: string | null;
}

/** One character with annotated images, ready to become a LoRA dataset. */
export interface IdentitySampleSummary {
  identity_id: string;
  display_name: string;
  label_name: string;
  image_count: number;
  manual_count: number;
}

export interface LoraExportCandidates {
  identities: IdentitySampleSummary[];
  default_output_dir: string;
}

export type LoraCropMode = "person" | "head" | "bust" | "large" | "none";
export type LoraQualityPreset = "none" | "pony" | "illustrious" | "sd15";

export interface LoraExportOptions {
  identity_id: string;
  trigger?: string | null;
  output_dir?: string | null;
  crop_mode?: LoraCropMode;
  resolution?: number;
  max_images?: number;
  duplicate_distance?: number;
  quality_preset?: LoraQualityPreset;
  extra_tags?: string;
  remove_tags?: string;
  drop_character_tags?: boolean;
  general_threshold?: number;
  character_threshold?: number;
  repeats?: number;
  keep_tokens?: number;
  shuffle_caption?: boolean;
  base_model?: string;
}

export interface LoraExportSample {
  source: string;
  image: string;
  caption: string;
  width: number;
  height: number;
}

export interface LoraExportResult {
  output_dir: string;
  train_dir: string;
  dataset_toml: string;
  kept: number;
  skipped: number;
  samples: LoraExportSample[];
  skip_reasons: Array<{ path: string; reason: string }>;
  trigger: string;
}

export type KohyaTrainerKind = "sd15" | "sdxl";

/** Detected state of the user's own kohya/sd-scripts installation. */
export interface KohyaStatus {
  configured: boolean;
  installed: boolean;
  root: string;
  sd_scripts_dir: string;
  python: string;
  python_source: string;
  trainers: string[];
  base_model: string;
  base_model_exists: boolean;
  train_data_dir: string;
  output_dir: string;
  error?: string | null;
  /** Command that would be executed, so the user can verify it first. */
  arguments_preview: string[];
}

export interface KohyaEnvironment {
  executable: string;
  python_version?: string | null;
  torch_version?: string | null;
  cuda_version?: string | null;
  cuda_available: boolean;
  device_name?: string | null;
  xformers_version?: string | null;
  bitsandbytes_version?: string | null;
  error?: string | null;
  exit_code?: number | null;
  stderr: string;
}

export interface KohyaTrainingRequest {
  dataset_config: string;
  output_dir: string;
  output_name: string;
  base_model: string;
  trainer: KohyaTrainerKind;
  network_dim: number;
  network_alpha: number;
  learning_rate: number;
  max_train_epochs: number;
  train_batch_size: number;
  gradient_accumulation_steps: number;
  optimizer: string;
  mixed_precision: string;
  save_every_n_epochs: number;
  seed: number;
  max_data_loader_n_workers: number;
  gradient_checkpointing: boolean;
  cache_latents: boolean;
  network_train_unet_only: boolean;
  /** Frees both CLIP encoders (~1.4 GB on SDXL); needs UNet-only training. */
  cache_text_encoder_outputs: boolean;
  flip_aug: boolean;
  noise_offset: number;
}

export interface KohyaTrainingStatus {
  running: boolean;
  finished: boolean;
  stopping: boolean;
  exit_code?: number | null;
  output_dir: string;
  output_name: string;
  base_model: string;
  dataset_config: string;
  log_path: string;
  started_at: number;
  elapsed_secs: number;
  epoch?: number | null;
  step?: number | null;
  total_steps?: number | null;
  loss?: number | null;
  message: string;
  artifacts: string[];
  log_tail: string[];
}

export interface KohyaInstallResult {
  destination: string;
  replaced: boolean;
}

export interface WorkerRuntimeSettings {
  mode: WorkerRuntimeMode;
  mode_label: string;
  resolved_path?: string | null;
  compute_device: WorkerComputeDevice;
  compute_device_label: string;
}

export type DatabaseStatus = "ready" | "initializing" | "error";

export interface IpcError {
  code: string;
  message: string;
  detail?: string;
  retryable: boolean;
  request_id: string;
  task_id?: string;
}

export interface IpcEnvelope<TPayload = unknown> {
  version: typeof IPC_SCHEMA_VERSION;
  request_id: string;
  task_id?: string;
  message_type: MessageType;
  payload?: TPayload;
  error?: IpcError;
}

export interface DatabaseHealth {
  status: DatabaseStatus;
  path: string;
  schema_version: number;
  table_count: number;
  message: string;
}

export interface WorkerHealth {
  status: WorkerStatus;
  message: string;
  version?: string;
  model_count?: number;
  runtime_mode?: WorkerRuntimeMode;
  runtime_mode_label?: string;
  runtime_path?: string | null;
  compute_device?: WorkerComputeDevice;
  compute_device_label?: string;
  capabilities?: WorkerCapabilities | null;
  capability_error?: string | null;
  checked_at: string;
}

export type WorkerCapabilityStatus = "ready" | "unavailable";

export interface WorkerCapabilityError {
  code: string;
  message: string;
  detail?: string | null;
  retryable: boolean;
}

export interface WorkerDependencyCapability {
  distribution: string;
  version?: string | null;
  status: WorkerCapabilityStatus;
  error?: WorkerCapabilityError | null;
}

export interface WorkerFeatureCapability {
  module: string;
  status: WorkerCapabilityStatus;
  error?: WorkerCapabilityError | null;
}

export interface WorkerComputeCapability {
  distribution?: string | null;
  available_providers: string[];
  /** The build advertises CUDA; it says nothing about the local runtime. */
  cuda_available: boolean;
  /** CUDA is advertised *and* the CUDA/cuDNN libraries load on this machine. */
  cuda_usable?: boolean;
  cuda_runtime?: WorkerCudaRuntime | null;
  error?: WorkerCapabilityError | null;
}

/** Why a CUDA-capable build can still end up running on CPU. */
export interface WorkerCudaRuntime {
  status: "ready" | "missing" | "unsupported";
  missing: string[];
  message: string;
  /** Folder the Worker was told to search, when one is configured. */
  configured_dir?: string | null;
}

/** Locally downloaded CUDA 12 / cuDNN 9 runtime inside `app\cuda`. */
export interface CudaRuntimeStatus {
  directory: string;
  installed: boolean;
  packages: string[];
  size_mb: number;
  message: string;
}

export interface CudaInstallProgress {
  phase: string;
  current: number;
  total: number;
  message: string;
}

/** An installed character recognizer as reported by the Worker. */
export interface InstalledRecognizerModel {
  id: string;
  kind: string;
  adapter: string;
  version: string;
  status: string;
  active: boolean;
  error?: string | null;
}

export interface RecognizerCatalogEntry {
  id: string;
  name: string;
  kind: string;
  repo_id: string;
  size_mb: number;
  license: string;
  note: string;
}

export interface RecognizerInventory {
  models: InstalledRecognizerModel[];
  active: string;
  catalog: RecognizerCatalogEntry[];
}

/** State of the reference-image library used for unlabelled characters. */
export interface ReferenceLibraryStatus {
  enabled: boolean;
  backend: string;
  path: string;
  exists: boolean;
  identities: number;
  references: number;
  built_at?: string | null;
  characters: Array<{
    identity_id: string;
    display_name: string;
    references: number;
  }>;
}

export interface WorkerCapabilities {
  status: "ok" | "unavailable";
  ready: boolean;
  dghs_imgutils: WorkerDependencyCapability;
  onnxruntime: WorkerDependencyCapability;
  compute?: WorkerComputeCapability | null;
  features: Record<string, WorkerFeatureCapability>;
  errors: WorkerCapabilityError[];
}

export interface RuntimeStatus {
  app_version: string;
  data_dir: string;
  database: DatabaseHealth;
  worker: WorkerHealth;
}

export interface GpuInventory {
  devices: string[];
  available: boolean;
  detail?: string | null;
}

export interface WorkerModelCompute {
  model_id: string;
  requested_device?: string | null;
  active_device?: string | null;
  active_providers: string[];
}

export interface WorkerComputeError {
  model_id: string;
  code?: string | null;
  error: string;
}

/**
 * Result of loading the installed models with the configured execution device.
 * `cuda_available` only means the ONNX Runtime build advertises CUDA, while
 * `cuda_session_ready` means a session really used the CUDA provider.
 */
export interface WorkerComputeStatus {
  requested_device: string;
  cuda_available: boolean;
  cuda_session_ready: boolean;
  distribution?: string | null;
  available_providers: string[];
  models: WorkerModelCompute[];
  errors: WorkerComputeError[];
}

export interface UpdateInfo {
  status: "available" | "up_to_date" | "unconfigured" | "unavailable";
  current_version: string;
  version?: string;
  date?: string;
  body?: string;
  message: string;
}

export interface UpdateProgress {
  phase: "downloading" | "installing" | "cancelled" | "complete";
  downloaded: number;
  total?: number;
  percent?: number;
  message: string;
}

export interface LibrarySelection {
  path: string;
  display_name: string;
}

/** Normalized coordinates used by editable image annotations. */
export interface NormalizedBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type AnnotationSource = "model" | "manual";

/** A persisted or user-edited annotation for one image. */
export interface ImageAnnotation {
  id?: string;
  identity_id?: string | null;
  label_name: string;
  bbox: NormalizedBoundingBox;
  source: AnnotationSource;
  manually_adjusted: boolean;
}

export interface ImageAnnotationsPayload {
  path: string;
  image_size: [number, number];
  /** Revision used for optimistic concurrency when saving annotations. */
  revision: number;
  annotations: ImageAnnotation[];
  learned_samples?: number;
  /** True when the saved labels changed the sample set enough to suggest a training run. */
  training_recommended?: boolean;
  updated_at?: string;
}

export type PersonalModelVersionStatus = "draft" | "active" | "archived" | "failed" | string;

/** Evaluation values returned by the local personal-model trainer. */
export type PersonalModelMetrics = Record<string, number | string | null | undefined>;

export interface PersonalModelVersion {
  id?: string;
  /** Numeric version assigned by the desktop core; the UI displays it as personal-vN. */
  version: number;
  status: PersonalModelVersionStatus;
  algorithm: string;
  sample_count: number;
  class_count: number;
  artifact?: unknown | null;
  metrics?: PersonalModelMetrics | null;
  warnings?: string[] | null;
  /** Only an eligible draft can be activated by the UI. */
  eligible_for_activation?: boolean;
  error_code?: string | null;
  error_message?: string | null;
  created_at: string;
  updated_at?: string | null;
  activated_at?: string | null;
  last_error?: string | null;
}

export interface TrainingClassProgress {
  identity_id: string;
  display_name: string;
  sample_count: number;
  min_required: number;
  eligible: boolean;
}

export interface PersonalTrainingReadiness {
  verified_sample_count: number;
  verified_class_count: number;
  eligible_class_count: number;
  ready: boolean;
  reasons: string[];
  class_progress?: TrainingClassProgress[];
}

/** Settings are owned by the core and may gain fields without breaking this contract. */
export type PersonalTrainingSettings = Record<string, unknown>;

/** Canonical nested status returned by get_personal_training_status and model actions. */
export interface PersonalTrainingStatus {
  status: string;
  message: string;
  readiness: PersonalTrainingReadiness;
  settings: PersonalTrainingSettings;
  active_version: number | null;
  versions: PersonalModelVersion[];
  updated_at?: string | null;
  last_error?: string | null;
  /** Optional legacy fields accepted while older desktop cores are being upgraded. */
  verified_sample_count?: number;
  class_count?: number;
  ready?: boolean;
  readiness_message?: string;
}

/** Optional request payloads are intentionally small so older cores can ignore extensions. */
export interface TrainPersonalModelPayload {
  force?: boolean;
  auto_activate?: boolean;
  overwrite_version?: number | null;
}

export interface ActivatePersonalModelPayload {
  version: number;
}

export interface DeletePersonalModelPayload {
  version: number;
}

export interface PersonalModelActionPayload {
  status?: PersonalTrainingStatus;
  version?: number | null;
  message?: string;
}

export interface CharacterIdentity {
  id: string;
  display_name: string;
  aliases?: string[];
  sample_count?: number;
}

export interface CharacterIdentityListPayload {
  identities: CharacterIdentity[];
}

/** Progress emitted by the desktop core while a bounded library preview runs. */
export interface LibraryScanProgress {
  phase: string;
  current: number;
  total: number;
  path?: string;
  message: string;
}

export interface LibraryScanBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
}

export interface LibraryScanCandidate {
  character_tag: string;
  display_name?: string | null;
  confidence: number;
  rank: number;
  crop_support: number;
  /** `base_model`, `personal_model`, or `fused` when personal learning is active. */
  source?: string | null;
  base_score?: number | null;
  personal_score?: number | null;
  sample_count?: number | null;
  reason?: string | null;
  embedding?: number[] | null;
  embedding_dimension?: number | null;
}

export type LibraryScanPersonStatus = "high_confidence" | "needs_review" | "unrecognized" | string;

export interface LibraryScanPerson {
  person_index: number;
  box: LibraryScanBox;
  crops: Record<string, unknown>;
  status: LibraryScanPersonStatus;
  top1?: LibraryScanCandidate | null;
  top2?: LibraryScanCandidate | null;
  margin: number;
  consistency: number;
  candidates: LibraryScanCandidate[];
  reason?: string | null;
  embedding?: number[] | null;
  embedding_dimension?: number | null;
}

export interface LibraryScanResult {
  path: string;
  image_size: [number, number];
  people: LibraryScanPerson[];
}

export interface LibraryScanError {
  path: string;
  code?: string;
  message: string;
  detail?: string | null;
}

/** Incremental result emitted after one library image has finished processing. */
export interface LibraryScanResultEvent {
  directory: string;
  total_discovered: number;
  processed: number;
  result?: LibraryScanResult | null;
  error?: LibraryScanError | null;
  model_version?: number | null;
  model_name?: string | null;
  skip_annotated?: boolean;
}

/** Bounded, read-only scan result returned by scan_library_preview. */
export interface LibraryScanPayload {
  directory: string;
  total_discovered: number;
  processed: number;
  cancelled: boolean;
  results: LibraryScanResult[];
  errors: LibraryScanError[];
  model_version?: number | null;
  model_name?: string;
  skip_annotated?: boolean;
  completed_at?: string;
}

export interface ScanLibraryArgs {
  path: string;
  limit?: number;
  model_version?: number | null;
  skip_annotated?: boolean;
}

export interface BatchMoveFilesPayload {
  sources: string[];
  destination_directory: string;
  subfolder_by_category?: string | null;
}

export interface BatchMoveResult {
  success_count: number;
  failed_count: number;
  moved: Array<{ source: string; destination: string }>;
  errors: Array<{ source: string; error: string }>;
}

export type SimilarityGroupType = "exact_duplicate" | "variation" | "similar";

export type SimilarityDecisionAction = "keep" | "delete" | "archive" | "unreviewed";

export interface SimilarityGroupItem {
  path: string;
  file_size: number;
  dimensions: [number, number];
  format: string;
  clarity_score: number;
  is_recommended: boolean;
  recommend_reason?: string | null;
  decision?: SimilarityDecisionAction;
  /** Total frames in the file (1 for still images). */
  frame_count?: number;
  /** True when the file is an animated GIF/WebP. */
  is_animated?: boolean;
  /** Frame indices that took part in the comparison. */
  sampled_frames?: number[];
}

export interface SimilarityGroup {
  group_id: string;
  group_type: SimilarityGroupType;
  average_similarity: number;
  items: SimilarityGroupItem[];
}

export interface SimilarityScanArgs {
  path: string;
  threshold?: number;
  include_subfolders?: boolean;
}

export interface SimilarityScanProgress {
  phase: string;
  current: number;
  total: number;
  path?: string;
  message: string;
  directory?: string | null;
}

export interface SimilarityScanState {
  running: boolean;
  directory?: string | null;
  phase: string;
  current: number;
  total: number;
  path?: string | null;
  message: string;
}

export interface SimilarityScanPayload {
  directory: string;
  total_scanned: number;
  groups: SimilarityGroup[];
  duplicates_count: number;
  potential_space_saved: number;
  cancelled: boolean;
  completed_at?: string;
}

export interface SimilarityItemDecision {
  path: string;
  action: SimilarityDecisionAction;
}

export interface SimilarityDecisionRequest {
  archive_directory?: string | null;
  archive_directory_name?: string | null;
  result_directory?: string | null;
  decisions: SimilarityItemDecision[];
}

export interface SimilarityDecisionResult {
  success_count: number;
  failed_count: number;
  processed: Array<{ path: string; action: SimilarityDecisionAction; destination?: string | null }>;
  errors: Array<{ path: string; error: string }>;
  freed_bytes: number;
}

export interface ApiResponse<TPayload> {
  envelope: IpcEnvelope<TPayload>;
  ok: boolean;
}

export function createRequest<TPayload>(
  message_type: MessageType,
  payload: TPayload,
  task_id?: string,
): IpcEnvelope<TPayload> {
  const request_id = globalThis.crypto?.randomUUID?.() ?? `req-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return {
    version: IPC_SCHEMA_VERSION,
    request_id,
    task_id,
    message_type,
    payload,
  };
}

export function isIpcEnvelope(value: unknown): value is IpcEnvelope {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    candidate.version === IPC_SCHEMA_VERSION &&
    typeof candidate.request_id === "string" &&
    typeof candidate.message_type === "string" &&
    (candidate.payload !== undefined || candidate.error !== undefined)
  );
}
