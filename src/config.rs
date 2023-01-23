use std::path::PathBuf;

use anyhow::{ensure, Result};
use clap::Parser;

#[derive(Clone, Debug, Parser)]
pub struct StorageConfig {
    /// Path to a directory containing swisssurface3d tifs
    #[clap(long, default_value = "/media/fl/DDLN-FL21/swisstopo/surface/conv/")]
    pub surface_dir: PathBuf,
    /// Path to a directory containing swissalti3d tifs
    #[clap(long, default_value = "/media/fl/DDLN-FL21/swisstopo/alti/conv/")]
    pub alti_dir: PathBuf,
    /// Path to a directory containing swissimage 10cm jpegs
    #[clap(long, default_value = "/media/fl/DDLN-FL21/swisstopo/image/conv/")]
    pub image_dir: PathBuf,
}

impl StorageConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.surface_dir.exists(),
            "Unable to access swisstopo surface model dir"
        );
        ensure!(
            self.alti_dir.exists(),
            "Unable to access swisstopo altitude model dir"
        );
        ensure!(
            self.image_dir.exists(),
            "Unable to access swisstopo ortho image dir"
        );
        Ok(())
    }
}
