plugins {
    id("com.android.application")
}

fun releaseVersionName(): String {
    val tag = System.getenv("GITHUB_REF_NAME")?.removePrefix("v")
    return tag?.takeIf { it.isNotBlank() } ?: "0.1.0"
}

android {
    namespace = "com.cranamp.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.cranamp.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = releaseVersionName()
    }

    buildTypes {
        debug {
            ndk {
                abiFilters.add("x86_64")
            }
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = signingConfigs.getByName("debug")

            ndk {
                abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86", "x86_64")
            }
        }
    }

    sourceSets {
        getByName("debug") {
            jniLibs.directories.add("../target/android")
        }
        getByName("release") {
            jniLibs.directories.add("../target/android")
        }
    }
}

fun checkCargoNdk() {
    val result = providers.exec {
        commandLine("cargo", "ndk", "--version")
        isIgnoreExitValue = true
    }.result.get()

    if (result.exitValue != 0) {
        throw GradleException(
            "cargo-ndk is not installed. Install it with: cargo install cargo-ndk"
        )
    }
}

tasks.register<Exec>("buildRustDebug") {
    description = "Build Cranamp Rust library for Android debug."
    group = "rust"

    doFirst {
        checkCargoNdk()
    }

    workingDir = rootProject.projectDir

    commandLine("sh", "-c", """
        cargo ndk \
            --platform 26 \
            -t x86_64 \
            -o target/android \
            build \
            --manifest-path ../../Cargo.toml \
            --lib \
            --features android,renderer-wgpu \
            --no-default-features
    """)
}

tasks.register<Exec>("buildRustRelease") {
    description = "Build Cranamp Rust library for Android release."
    group = "rust"

    doFirst {
        checkCargoNdk()
    }

    workingDir = rootProject.projectDir

    commandLine("sh", "-c", """
        cargo ndk \
            --platform 26 \
            -t arm64-v8a \
            -t armeabi-v7a \
            -t x86 \
            -t x86_64 \
            -o target/android \
            build \
            --release \
            --manifest-path ../../Cargo.toml \
            --lib \
            --features android,renderer-wgpu \
            --no-default-features
    """)
}

afterEvaluate {
    tasks.matching { it.name.startsWith("merge") && it.name.contains("NativeLibs") }.configureEach {
        if (name.contains("Debug", ignoreCase = true)) {
            dependsOn("buildRustDebug")
        } else if (name.contains("Release", ignoreCase = true)) {
            dependsOn("buildRustRelease")
        }
    }
}
