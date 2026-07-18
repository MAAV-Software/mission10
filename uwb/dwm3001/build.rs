fn main() {
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set"));
    std::fs::copy("memory.x", output.join("memory.x")).expect("copy memory.x");
    println!("cargo:rustc-link-search={}", output.display());
    println!("cargo:rerun-if-changed=memory.x");
}
