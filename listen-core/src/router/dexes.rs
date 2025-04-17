use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DexName {
    Orca,
    Raydium,
    Jupiter,
    Unknown,
}

impl fmt::Display for DexName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DexName::Orca => write!(f, "Orca"),
            DexName::Raydium => write!(f, "Raydium"),
            DexName::Jupiter => write!(f, "Jupiter"),
            DexName::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Dex {
    pub name: DexName,
    pub program_id: String,
}

impl Dex {
    pub fn name(&self) -> DexName {
        self.name
    }

    pub fn program_id(&self) -> &str {
        &self.program_id
    }
}
