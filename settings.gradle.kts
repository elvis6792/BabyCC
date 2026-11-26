// Fix conflicting ANDROID_PREFS_ROOT env var passed by host environment
try {
    val processEnvironment = Class.forName("java.lang.ProcessEnvironment")
    val theEnvironmentField = processEnvironment.getDeclaredField("theEnvironment")
    theEnvironmentField.isAccessible = true
    @Suppress("UNCHECKED_CAST")
    val env = theEnvironmentField.get(null) as MutableMap<String, String>
    env.remove("ANDROID_PREFS_ROOT")

    val theCaseInsensitiveEnvironmentField = processEnvironment.getDeclaredField("theCaseInsensitiveEnvironment")
    theCaseInsensitiveEnvironmentField.isAccessible = true
    @Suppress("UNCHECKED_CAST")
    val cienv = theCaseInsensitiveEnvironmentField.get(null) as MutableMap<String, String>
    cienv.remove("ANDROID_PREFS_ROOT")
} catch (e: Throwable) {
    // Ignore
}

pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "BabyCC-Cough-Analyzer"
include(":app")
