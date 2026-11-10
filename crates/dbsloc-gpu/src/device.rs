// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::capacity::DeviceMemory;
#[cfg(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd"
))]
use crate::capacity::MemoryFigure;
use crate::caps::GpuCaps;
use crate::error::{GpuError, GpuResult};
use crate::profiler::Profiler;

/// Backend requested for device selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BackendChoice {
    Auto,
    Vulkan,
    Dx12,
    Metal,
}

impl BackendChoice {
    /// Parses `auto|vulkan|dx12|metal`.
    pub fn parse(s: &str) -> GpuResult<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(BackendChoice::Auto),
            "vulkan" | "vk" => Ok(BackendChoice::Vulkan),
            "dx12" | "d3d12" => Ok(BackendChoice::Dx12),
            "metal" => Ok(BackendChoice::Metal),
            other => Err(GpuError::Invalid(format!(
                "E_GPU_BACKEND: unknown graphics API {other:?} (known: auto, vulkan, dx12, metal)"
            ))),
        }
    }

    fn candidates(self) -> Vec<wgpu::Backends> {
        match self {
            BackendChoice::Vulkan => vec![wgpu::Backends::VULKAN],
            BackendChoice::Dx12 => vec![wgpu::Backends::DX12],
            BackendChoice::Metal => vec![wgpu::Backends::METAL],
            BackendChoice::Auto => {
                if cfg!(target_os = "macos") {
                    vec![wgpu::Backends::METAL]
                } else if cfg!(target_os = "windows") {
                    vec![wgpu::Backends::VULKAN, wgpu::Backends::DX12]
                } else {
                    vec![wgpu::Backends::VULKAN]
                }
            }
        }
    }
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Adapter selection request.
#[derive(Debug, Clone)]
pub struct BackendSelect {
    pub backend: BackendChoice,
    pub adapter_substring: Option<String>,
    pub allow_software: bool,
    pub force_software: bool,
}

impl Default for BackendSelect {
    fn default() -> Self {
        BackendSelect {
            backend: BackendChoice::Auto,
            adapter_substring: None,
            allow_software: false,
            force_software: false,
        }
    }
}

/// Environment variable that selects the graphics API for adapter selection.
pub const GPU_BACKEND_ENV: &str = "DBSLOC_GPU_BACKEND";

impl BackendSelect {
    /// Reads `DBSLOC_GPU_BACKEND`, `DBSLOC_GPU_ADAPTER`, `DBSLOC_GPU_ALLOW_SOFTWARE` and
    /// `WGPU_FORCE_FALLBACK_ADAPTER`. An invalid `DBSLOC_GPU_BACKEND` is an error listing the
    /// valid values; blank is `auto`.
    pub fn from_env() -> GpuResult<Self> {
        let backend = match std::env::var(GPU_BACKEND_ENV) {
            Ok(v) if !v.trim().is_empty() => BackendChoice::parse(&v).map_err(|e| match e {
                GpuError::Invalid(m) => GpuError::Invalid(format!("{GPU_BACKEND_ENV}: {m}")),
                other => other,
            })?,
            _ => BackendChoice::Auto,
        };
        let adapter_substring = std::env::var("DBSLOC_GPU_ADAPTER")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let force_software = env_truthy("WGPU_FORCE_FALLBACK_ADAPTER");
        Ok(BackendSelect {
            backend,
            adapter_substring,
            allow_software: force_software || env_truthy("DBSLOC_GPU_ALLOW_SOFTWARE"),
            force_software,
        })
    }
}

/// Identity of the selected adapter.
#[derive(Debug, Clone, Serialize)]
pub struct AdapterRecord {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub driver: String,
    pub driver_info: String,
    pub backend: String,
}

impl AdapterRecord {
    /// Builds a record from adapter info.
    pub fn from_info(info: &wgpu::AdapterInfo) -> Self {
        AdapterRecord {
            name: info.name.clone(),
            vendor: info.vendor,
            device: info.device,
            device_type: format!("{:?}", info.device_type),
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
            backend: format!("{:?}", info.backend),
        }
    }

    /// Whether the adapter is a software rasteriser.
    pub fn is_software(&self) -> bool {
        self.device_type == "Cpu"
    }
}

/// Creates an instance restricted to `backends`.
///
/// The first call whose `backends` include Vulkan also creates one idle Vulkan instance that is
/// kept for the life of the process (no device, no GPU memory), see `keep_vulkan_loaded`.
pub fn new_instance(backends: wgpu::Backends) -> wgpu::Instance {
    if backends.contains(wgpu::Backends::VULKAN) {
        keep_vulkan_loaded();
    }
    create_instance(backends)
}

fn create_instance(backends: wgpu::Backends) -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::from_env_or_default(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

#[cfg(not(target_os = "macos"))]
static KEEP_VULKAN: std::sync::OnceLock<wgpu::Instance> = std::sync::OnceLock::new();

/// One idle Vulkan instance kept for the life of the process (no device, no GPU memory).
///
/// The engine creates and destroys a Vulkan instance for every device it opens. Each destroy lets
/// the Vulkan loader unload the driver library, and NVIDIA's (`libGLX_nvidia.so.0`) cannot be
/// loaded again indefinitely: measured on Linux with driver 580, the 11th instance in a process
/// gets no `vkCreateInstance` from it ("loader_scanned_icd_add: Could not get 'vkCreateInstance'"),
/// so the GPU vanishes for every later call -- a cohort run fell back after ten subjects. With
/// this instance alive the loader and the driver stay loaded and later instances reuse them.
/// macOS runs Metal, which has no loader.
#[cfg(not(target_os = "macos"))]
fn keep_vulkan_loaded() {
    KEEP_VULKAN.get_or_init(|| create_instance(wgpu::Backends::VULKAN));
}

#[cfg(target_os = "macos")]
fn keep_vulkan_loaded() {}

/// Lists the adapters of `instance` for `backends`.
pub fn enumerate(instance: &wgpu::Instance, backends: wgpu::Backends) -> Vec<wgpu::Adapter> {
    pollster::block_on(instance.enumerate_adapters(backends))
}

fn type_rank(t: wgpu::DeviceType) -> u32 {
    match t {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    }
}

fn pick(adapters: Vec<wgpu::Adapter>, sel: &BackendSelect) -> Option<wgpu::Adapter> {
    let needle = sel
        .adapter_substring
        .as_ref()
        .map(|s| s.to_ascii_lowercase());
    let mut best: Option<(u32, wgpu::Adapter)> = None;
    for a in adapters {
        let info = a.get_info();
        let software = info.device_type == wgpu::DeviceType::Cpu;
        if sel.force_software && !software {
            continue;
        }
        if software && !sel.allow_software {
            continue;
        }
        if let Some(n) = &needle {
            if !info.name.to_ascii_lowercase().contains(n) {
                continue;
            }
        }
        let rank = type_rank(info.device_type);
        if best.as_ref().map(|(r, _)| rank < *r).unwrap_or(true) {
            best = Some((rank, a));
        }
    }
    best.map(|(_, a)| a)
}

/// Limits requested from the adapter: defaults clamped to the adapter, with the compute-related
/// limits raised to what the adapter offers.
pub fn required_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let a = adapter.limits();
    let mut l = wgpu::Limits::default().or_worse_values_from(&a);
    l.max_storage_buffer_binding_size = a.max_storage_buffer_binding_size;
    l.max_buffer_size = a.max_buffer_size;
    l.max_compute_workgroup_storage_size = a.max_compute_workgroup_storage_size;
    l.max_compute_invocations_per_workgroup = a.max_compute_invocations_per_workgroup;
    l.max_compute_workgroup_size_x = a.max_compute_workgroup_size_x;
    l.max_compute_workgroup_size_y = a.max_compute_workgroup_size_y;
    l.max_compute_workgroup_size_z = a.max_compute_workgroup_size_z;
    l.max_compute_workgroups_per_dimension = a.max_compute_workgroups_per_dimension;
    l.max_storage_buffers_per_shader_stage = a.max_storage_buffers_per_shader_stage;
    l
}

/// Optional features requested when the adapter has them.
pub fn optional_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    adapter.features() & wgpu::Features::TIMESTAMP_QUERY
}

/// Requests a device with the required limits and optional features; returns the error text of a
/// failed request verbatim.
pub fn request_device(
    adapter: &wgpu::Adapter,
) -> GpuResult<(wgpu::Device, wgpu::Queue, wgpu::Features)> {
    let features = optional_features(adapter);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("dbsloc-gpu"),
        required_features: features,
        required_limits: required_limits(adapter),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .map_err(|e| GpuError::RequestDevice(e.to_string()))?;
    Ok((device, queue, features))
}

/// Selected adapter with its device, queue, capabilities and profiler.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: AdapterRecord,
    pub caps: GpuCaps,
    pub memory: Option<DeviceMemory>,
    pub features: wgpu::Features,
    pub(crate) profiler: Profiler,
    errors: Arc<Mutex<Vec<String>>>,
    lost: Arc<AtomicBool>,
}

impl GpuContext {
    /// Selects an adapter according to `sel` and creates the device.
    pub fn new(sel: &BackendSelect) -> GpuResult<Self> {
        Self::new_with(sel, Ok)
    }

    /// Like [`Self::new`], then hands the context to `accept` (e.g. a known-answer self-test).
    /// When creating the device or `accept` fails on one graphics API, the next candidate API is
    /// tried; the error lists every attempt when all of them fail.
    pub fn new_with<T>(
        sel: &BackendSelect,
        mut accept: impl FnMut(GpuContext) -> GpuResult<T>,
    ) -> GpuResult<T> {
        let mut tried = Vec::new();
        let mut failures = Vec::new();
        for backends in sel.backend.candidates() {
            let instance = new_instance(backends);
            let adapters = enumerate(&instance, backends);
            let count = adapters.len();
            match pick(adapters, sel) {
                Some(adapter) => {
                    let name = adapter.get_info().name;
                    match Self::from_adapter(instance, adapter).and_then(&mut accept) {
                        Ok(v) => return Ok(v),
                        Err(e) => {
                            tried.push(format!("{backends:?}: {name}: {e}"));
                            failures.push(e);
                        }
                    }
                }
                None => tried.push(format!("{backends:?}: {count} adapter(s), none eligible")),
            }
        }
        if tried.len() == 1 && failures.len() == 1 {
            if let Some(e) = failures.pop() {
                return Err(e);
            }
        }
        let mut why = tried.join("; ");
        if let Some(first) = failures.first() {
            return Err(combined_failure(first, why));
        }
        if let Some(s) = &sel.adapter_substring {
            why.push_str(&format!("; adapter filter '{s}'"));
        }
        if sel.force_software {
            why.push_str("; WGPU_FORCE_FALLBACK_ADAPTER set (software adapters only)");
        } else if !sel.allow_software {
            why.push_str("; software adapters not allowed (DBSLOC_GPU_ALLOW_SOFTWARE=1)");
        }
        Err(GpuError::NoAdapter(why))
    }

    fn from_adapter(instance: wgpu::Instance, adapter: wgpu::Adapter) -> GpuResult<Self> {
        let info = AdapterRecord::from_info(&adapter.get_info());
        let (device, queue, features) = request_device(&adapter)?;
        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = errors.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            sink.lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(e.to_string());
        }));
        let lost_sink = errors.clone();
        let lost = Arc::new(AtomicBool::new(false));
        let lost_flag = lost.clone();
        device.set_device_lost_callback(move |reason, msg| {
            lost_flag.store(true, Ordering::SeqCst);
            lost_sink
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(format!("device lost ({reason:?}): {msg}"));
        });
        let mut caps = GpuCaps::from_adapter(&adapter);
        caps.timestamp_period_ns = Some(queue.get_timestamp_period());
        let profiler = Profiler::new(&device, features);
        let memory = device_memory(&adapter);
        Ok(GpuContext {
            instance,
            adapter,
            device,
            queue,
            info,
            caps,
            memory,
            features,
            profiler,
            errors,
            lost,
        })
    }

    /// Whether the driver reported the device lost (a reset or removal); a lost device never
    /// recovers and must be replaced.
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }

    /// Marks the device lost as the device-lost callback does, for tests of the replacement path.
    #[doc(hidden)]
    pub fn mark_lost(&self) {
        self.lost.store(true, Ordering::SeqCst);
    }

    /// Drains errors reported outside error scopes; returns them as one error when present.
    pub fn check_errors(&self) -> GpuResult<()> {
        let mut guard = self.errors.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_empty() {
            return Ok(());
        }
        let text = guard.join(" | ");
        guard.clear();
        if text.contains("device lost") {
            Err(GpuError::DeviceLost(text))
        } else {
            Err(GpuError::Validation(text))
        }
    }

    /// Blocks until all submitted work has finished.
    pub fn sync(&self) -> GpuResult<()> {
        let timeout = wait_timeout();
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(timeout),
            })
            .map_err(|e| match e {
                wgpu::PollError::Timeout => GpuError::Internal(format!(
                    "GPU wait timed out after {} s (DBSLOC_GPU_WAIT_TIMEOUT_S)",
                    timeout.as_secs_f64()
                )),
                other => GpuError::Internal(format!("poll failed: {other}")),
            })?;
        self.check_errors()
    }

    /// Runs `f` inside out-of-memory, validation and internal error scopes and converts a
    /// captured error into a [`GpuError`].
    pub fn scoped<T>(&self, label: &str, requested: u64, f: impl FnOnce() -> T) -> GpuResult<T> {
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let val = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let int = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let out = f();
        let ie = pollster::block_on(int.pop());
        let ve = pollster::block_on(val.pop());
        let oe = pollster::block_on(oom.pop());
        if oe.is_some() {
            return Err(GpuError::OutOfMemory {
                requested,
                label: label.to_string(),
            });
        }
        if let Some(e) = ve {
            return Err(GpuError::Validation(format!("{label}: {e}")));
        }
        if let Some(e) = ie {
            return Err(GpuError::Internal(format!("{label}: {e}")));
        }
        Ok(out)
    }
}

fn device_memory(adapter: &wgpu::Adapter) -> Option<DeviceMemory> {
    #[cfg(windows)]
    if let Some(m) = dx12_memory(adapter) {
        return Some(m);
    }
    #[cfg(any(
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    ))]
    if let Some(m) = vulkan_memory(adapter) {
        return Some(m);
    }
    let _ = adapter;
    None
}

#[cfg(windows)]
fn dx12_memory(adapter: &wgpu::Adapter) -> Option<DeviceMemory> {
    // SAFETY: read-only query on the live hal adapter; the adapter is never destroyed or modified while borrowed.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Dx12>() }?;
    let info = hal
        .raw_adapter()
        .query_video_memory_info(Default::default())
        .ok()?;
    (info.Budget > 0).then_some(DeviceMemory {
        bytes: info.Budget,
        figure: MemoryFigure::DxgiLocalBudget,
    })
}

#[cfg(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd"
))]
fn vulkan_memory(adapter: &wgpu::Adapter) -> Option<DeviceMemory> {
    const VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: u32 = 0x1;
    // SAFETY: read-only query on the live hal adapter; the adapter is never destroyed or modified while borrowed.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    // SAFETY: the physical device handle comes from the live instance that owns it; the call only reads its properties.
    let props = unsafe {
        hal.shared_instance()
            .raw_instance()
            .get_physical_device_memory_properties(hal.raw_physical_device())
    };
    let count = usize::try_from(props.memory_heap_count).ok()?;
    props
        .memory_heaps
        .get(..count)?
        .iter()
        .filter(|h| h.flags.as_raw() & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0)
        .map(|h| h.size)
        .max()
        .filter(|b| *b > 0)
        .map(|bytes| DeviceMemory {
            bytes,
            figure: MemoryFigure::VulkanDeviceLocalHeap,
        })
}

fn combined_failure(kind: &GpuError, text: String) -> GpuError {
    match kind {
        GpuError::NoAdapter(_) => GpuError::NoAdapter(text),
        GpuError::Validation(_) => GpuError::Validation(text),
        GpuError::Internal(_) => GpuError::Internal(text),
        GpuError::DeviceLost(_) => GpuError::DeviceLost(text),
        GpuError::Readback(_) => GpuError::Readback(text),
        GpuError::Invalid(_) => GpuError::Invalid(text),
        GpuError::Verification(_) => GpuError::Verification(text),
        GpuError::RequestDevice(_)
        | GpuError::OutOfMemory { .. }
        | GpuError::LimitsInsufficient { .. }
        | GpuError::Cancelled => GpuError::RequestDevice(text),
    }
}

const DEFAULT_WAIT_TIMEOUT_S: f64 = 300.0;

fn wait_timeout() -> std::time::Duration {
    let seconds = std::env::var("DBSLOC_GPU_WAIT_TIMEOUT_S")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_WAIT_TIMEOUT_S);
    std::time::Duration::try_from_secs_f64(seconds)
        .unwrap_or(std::time::Duration::from_secs_f64(DEFAULT_WAIT_TIMEOUT_S))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn new_instance_with_vulkan_keeps_an_idle_vulkan_instance() {
        drop(new_instance(wgpu::Backends::VULKAN));
        assert!(KEEP_VULKAN.get().is_some());
    }

    #[test]
    fn gpu_backend_env_is_parsed_and_an_invalid_value_is_an_error() {
        std::env::remove_var(GPU_BACKEND_ENV);
        assert_eq!(
            BackendSelect::from_env().unwrap().backend,
            BackendChoice::Auto
        );
        std::env::set_var(GPU_BACKEND_ENV, "  ");
        assert_eq!(
            BackendSelect::from_env().unwrap().backend,
            BackendChoice::Auto
        );
        std::env::set_var(GPU_BACKEND_ENV, "dx12");
        assert_eq!(
            BackendSelect::from_env().unwrap().backend,
            BackendChoice::Dx12
        );
        std::env::set_var(GPU_BACKEND_ENV, "cuda");
        let e = BackendSelect::from_env().unwrap_err().to_string();
        std::env::remove_var(GPU_BACKEND_ENV);
        assert!(e.contains(GPU_BACKEND_ENV), "{e}");
        assert!(e.contains("E_GPU_BACKEND"), "{e}");
        for v in ["auto", "vulkan", "dx12", "metal"] {
            assert!(e.contains(v), "{e}");
        }
    }
}
