plugins {
    id("com.android.application")
    kotlin("android")
}

val configuredModelManifestUrl = providers.gradleProperty("intelhive.modelManifestUrl")
    .orElse(
        "https://uozyxansakogtpqxcpdp.supabase.co/storage/v1/object/public/" +
            "model-artifacts/manifests/qwen2.5-3b-instruct/1.0.0/manifest.json"
    )
    .get()
    .replace("\\", "\\\\")
    .replace("\"", "\\\"")

val configuredSupabaseUrl = providers.gradleProperty("intelhive.supabaseUrl")
    .orElse("https://uozyxansakogtpqxcpdp.supabase.co")
    .get()
    .replace("\\", "\\\\")
    .replace("\"", "\\\"")

val configuredSupabasePublishableKey = providers.gradleProperty("intelhive.supabasePublishableKey")
    .orElse("sb_publishable_tmCxtqxpIaNKrimWVidtxA_BnULyxkK")
    .get()
    .replace("\\", "\\\\")
    .replace("\"", "\\\"")

android {
    namespace = "com.intellihive.worker"
    compileSdk = 34
    ndkVersion = "26.3.11579264"

    sourceSets["main"].java.srcDir("src/main/kotlin")

    defaultConfig {
        applicationId = "com.intellihive.worker"
        minSdk = 28
        targetSdk = 34
        versionCode = 8
        versionName = "0.1.7"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "MODEL_MANIFEST_URL", "\"$configuredModelManifestUrl\"")
        buildConfigField("String", "SUPABASE_URL", "\"$configuredSupabaseUrl\"")
        buildConfigField(
            "String",
            "SUPABASE_PUBLISHABLE_KEY",
            "\"$configuredSupabasePublishableKey\""
        )

        ndk {
            abiFilters += "arm64-v8a"
        }

        buildFeatures {
            buildConfig = true
        }

        externalNativeBuild {
            cmake {
                arguments += "-DINTELHIVE_LLAMA_SOURCE_DIR=" + providers.gradleProperty("intelhive.llamaSourceDir")
                    .orElse(rootProject.file("../.devtools/llama-source").absolutePath).get()
                cppFlags += "-std=c++17"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("com.squareup.okhttp3:okhttp:4.11.0")
    implementation("com.google.code.gson:gson:2.10.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:core:1.5.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
