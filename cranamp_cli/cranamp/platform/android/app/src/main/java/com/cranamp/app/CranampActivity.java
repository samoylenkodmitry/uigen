package com.cranamp.app;

import android.app.NativeActivity;
import android.content.ActivityNotFoundException;
import android.content.ContentResolver;
import android.content.Intent;
import android.database.Cursor;
import android.os.Build;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.provider.DocumentsContract;
import android.provider.OpenableColumns;
import android.util.Log;
import android.view.View;
import android.view.WindowInsets;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

public class CranampActivity extends NativeActivity {
    private static final String TAG = "CranampActivity";

    private static final int MODE_REPLACE = 0;
    private static final int MODE_APPEND = 1;
    private static final int REQ_AUDIO_REPLACE = 1001;
    private static final int REQ_AUDIO_APPEND = 1002;
    private static final int REQ_FOLDER_REPLACE = 1003;
    private static final int REQ_FOLDER_APPEND = 1004;
    private static final int REQ_IMPORT_PLAYLIST = 1005;
    private static final int REQ_EXPORT_PLAYLIST = 1006;
    private static final int REQ_IMPORT_SKIN = 1007;

    private static boolean moveUnavailableLogged = false;

    private String pendingExportText = "";

    private static final Set<String> AUDIO_EXTENSIONS = new HashSet<>();

    static {
        Collections.addAll(
                AUDIO_EXTENSIONS,
                "aac", "aiff", "alac", "caf", "flac", "m4a", "m4b", "m4v", "mov",
                "mp1", "mp2", "mp3", "mp4", "oga", "ogg", "opus", "wav", "wave", "webm"
        );
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ensureBridgeDir();
        ensureMediaDir();
        ensureSkinDir();
    }

    public String cranampBridgeDirectory() {
        return ensureBridgeDir().getAbsolutePath();
    }

    public boolean cranampStartWindowMove(float localXDp, float localYDp) {
        return invokeStartWindowMove(localXDp, localYDp);
    }

    public float cranampContentTopInsetDp() {
        int topPx = contentTopInsetPx();
        float density = getResources().getDisplayMetrics().density;
        return density > 0.0f ? topPx / density : topPx;
    }

    public float cranampContentBottomInsetDp() {
        int bottomPx = contentBottomInsetPx();
        float density = getResources().getDisplayMetrics().density;
        return density > 0.0f ? bottomPx / density : bottomPx;
    }

    public void cranampPickAudioFiles(int mode) {
        final int requestCode = mode == MODE_REPLACE ? REQ_AUDIO_REPLACE : REQ_AUDIO_APPEND;
        final String resultName = mode == MODE_REPLACE ? "audio_replace" : "audio_append";
        runOnUiThread(() -> {
            clearResult(resultName);
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("audio/*");
            intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            launchIntent(intent, requestCode, resultName);
        });
    }

    public void cranampPickAudioFolder(int mode) {
        final int requestCode = mode == MODE_REPLACE ? REQ_FOLDER_REPLACE : REQ_FOLDER_APPEND;
        final String resultName = mode == MODE_REPLACE ? "audio_replace" : "audio_append";
        runOnUiThread(() -> {
            clearResult(resultName);
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            intent.addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
            launchIntent(intent, requestCode, resultName);
        });
    }

    public void cranampImportPlaylist() {
        runOnUiThread(() -> {
            clearResult("playlist_import");
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("*/*");
            intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{
                    "audio/x-mpegurl",
                    "application/vnd.apple.mpegurl",
                    "application/x-mpegurl",
                    "text/plain"
            });
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            launchIntent(intent, REQ_IMPORT_PLAYLIST, "playlist_import");
        });
    }

    public void cranampExportPlaylist(String playlistText) {
        runOnUiThread(() -> {
            pendingExportText = playlistText == null ? "" : playlistText;
            clearResult("playlist_export");
            Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("audio/x-mpegurl");
            intent.putExtra(Intent.EXTRA_TITLE, "playlist.m3u");
            intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            launchIntent(intent, REQ_EXPORT_PLAYLIST, "playlist_export");
        });
    }

    public void cranampPickSkinFile() {
        runOnUiThread(() -> {
            clearResult("skin_import");
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("*/*");
            intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{
                    "application/zip",
                    "application/octet-stream",
                    "application/x-zip-compressed"
            });
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            launchIntent(intent, REQ_IMPORT_SKIN, "skin_import");
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null) {
            writeCancel(resultNameForRequest(requestCode));
            return;
        }

        try {
            switch (requestCode) {
                case REQ_AUDIO_REPLACE:
                    writePickedAudioUris("audio_replace", collectAudioUris(data));
                    break;
                case REQ_AUDIO_APPEND:
                    writePickedAudioUris("audio_append", collectAudioUris(data));
                    break;
                case REQ_FOLDER_REPLACE:
                    writePickedAudioUris("audio_replace", collectFolderAudioUris(data.getData()));
                    break;
                case REQ_FOLDER_APPEND:
                    writePickedAudioUris("audio_append", collectFolderAudioUris(data.getData()));
                    break;
                case REQ_IMPORT_PLAYLIST:
                    writePlaylistImport(data.getData());
                    break;
                case REQ_EXPORT_PLAYLIST:
                    writePlaylistExport(data.getData());
                    break;
                case REQ_IMPORT_SKIN:
                    writeSkinImport(data.getData());
                    break;
                default:
                    break;
            }
        } catch (Exception error) {
            writeError(resultNameForRequest(requestCode), error.toString());
        }
    }

    private void launchIntent(Intent intent, int requestCode, String resultName) {
        try {
            startActivityForResult(intent, requestCode);
        } catch (ActivityNotFoundException error) {
            writeError(resultName, "No Android document picker is available");
        }
    }

    private boolean invokeStartWindowMove(float localXDp, float localYDp) {
        try {
            View decor = getWindow().getDecorView();
            int[] location = new int[2];
            decor.getLocationOnScreen(location);
            float density = getResources().getDisplayMetrics().density;
            Rect insets = surfaceInsets();
            float rawX = location[0] + localXDp * density - insets.left;
            float rawY = location[1] + localYDp * density - insets.top;
            return startMovingTask(decor, rawX, rawY);
        } catch (Exception error) {
            if (!moveUnavailableLogged) {
                moveUnavailableLogged = true;
                Log.w(
                        TAG,
                        "Android freeform task moving is unavailable; overlay surface support is required: "
                                + error
                );
            }
            return false;
        }
    }

    private Rect surfaceInsets() {
        try {
            Field field = getWindow().getAttributes().getClass().getField("surfaceInsets");
            Object value = field.get(getWindow().getAttributes());
            if (value instanceof Rect) {
                return (Rect) value;
            }
        } catch (Exception ignored) {
        }
        return new Rect();
    }

    private int contentTopInsetPx() {
        View decor = getWindow().getDecorView();
        if (decor == null) {
            return 0;
        }

        int top = rootWindowTopInsetPx(decor);
        Rect visibleFrame = new Rect();
        decor.getWindowVisibleDisplayFrame(visibleFrame);
        int[] location = new int[2];
        decor.getLocationOnScreen(location);
        top = Math.max(top, visibleFrame.top - location[1]);
        top += Math.max(surfaceInsets().top, 0);
        top = Math.max(top, androidSystemDimensionPx("status_bar_height", 24));
        return Math.max(top, 0);
    }

    private int rootWindowTopInsetPx(View decor) {
        WindowInsets insets = decor.getRootWindowInsets();
        if (insets == null) {
            return 0;
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            int top = insets.getInsets(WindowInsets.Type.captionBar()).top;
            top = Math.max(top, insets.getInsets(WindowInsets.Type.systemBars()).top);
            top = Math.max(top, insets.getInsets(WindowInsets.Type.displayCutout()).top);
            return top;
        }
        return insets.getSystemWindowInsetTop();
    }

    private int contentBottomInsetPx() {
        View decor = getWindow().getDecorView();
        if (decor == null) {
            return 0;
        }

        int bottom = rootWindowBottomInsetPx(decor);
        Rect visibleFrame = new Rect();
        decor.getWindowVisibleDisplayFrame(visibleFrame);
        int[] location = new int[2];
        decor.getLocationOnScreen(location);
        int decorBottom = location[1] + decor.getHeight();
        bottom = Math.max(bottom, decorBottom - visibleFrame.bottom);
        bottom += Math.max(surfaceInsets().bottom, 0);
        bottom = Math.max(bottom, androidSystemDimensionPx("navigation_bar_height", 0));
        return Math.max(bottom, 0);
    }

    private int rootWindowBottomInsetPx(View decor) {
        WindowInsets insets = decor.getRootWindowInsets();
        if (insets == null) {
            return 0;
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            int bottom = insets.getInsets(WindowInsets.Type.systemBars()).bottom;
            bottom = Math.max(bottom, insets.getInsets(WindowInsets.Type.displayCutout()).bottom);
            return bottom;
        }
        return insets.getSystemWindowInsetBottom();
    }

    private int androidSystemDimensionPx(String name, int fallbackDp) {
        int id = getResources().getIdentifier(name, "dimen", "android");
        if (id > 0) {
            int value = getResources().getDimensionPixelSize(id);
            if (value > 0) {
                return value;
            }
        }
        float density = getResources().getDisplayMetrics().density;
        return Math.round(fallbackDp * density);
    }

    private boolean startMovingTask(View decor, float rawX, float rawY) throws Exception {
        try {
            Method method = View.class.getMethod("startMovingTask", float.class, float.class);
            Object result = method.invoke(decor, rawX, rawY);
            return result instanceof Boolean && (Boolean) result;
        } catch (NoSuchMethodException publicLookupFailed) {
            try {
                Method method = View.class.getDeclaredMethod("startMovingTask", float.class, float.class);
                method.setAccessible(true);
                Object result = method.invoke(decor, rawX, rawY);
                return result instanceof Boolean && (Boolean) result;
            } catch (NoSuchMethodException declaredLookupFailed) {
                return startMovingTaskViaWindowSession(decor, rawX, rawY);
            }
        }
    }

    private boolean startMovingTaskViaWindowSession(View decor, float rawX, float rawY) throws Exception {
        Field attachInfoField = View.class.getDeclaredField("mAttachInfo");
        attachInfoField.setAccessible(true);
        Object attachInfo = attachInfoField.get(decor);
        if (attachInfo == null) {
            return false;
        }

        Field sessionField = attachInfo.getClass().getDeclaredField("mSession");
        sessionField.setAccessible(true);
        Object session = sessionField.get(attachInfo);

        Field windowField = attachInfo.getClass().getDeclaredField("mWindow");
        windowField.setAccessible(true);
        Object window = windowField.get(attachInfo);

        if (session == null || window == null) {
            return false;
        }

        Method method = session.getClass().getMethod(
                "startMovingTask",
                Class.forName("android.view.IWindow"),
                float.class,
                float.class
        );
        Object result = method.invoke(session, window, rawX, rawY);
        return result instanceof Boolean && (Boolean) result;
    }

    private ArrayList<Uri> collectAudioUris(Intent data) {
        ArrayList<Uri> uris = new ArrayList<>();
        if (data.getClipData() != null) {
            for (int i = 0; i < data.getClipData().getItemCount(); i++) {
                Uri uri = data.getClipData().getItemAt(i).getUri();
                if (uri != null) {
                    uris.add(uri);
                }
            }
        } else if (data.getData() != null) {
            uris.add(data.getData());
        }
        return uris;
    }

    private ArrayList<Uri> collectFolderAudioUris(Uri treeUri) {
        ArrayList<Uri> uris = new ArrayList<>();
        if (treeUri == null) {
            return uris;
        }
        int flags = Intent.FLAG_GRANT_READ_URI_PERMISSION;
        try {
            getContentResolver().takePersistableUriPermission(treeUri, flags);
        } catch (SecurityException ignored) {
        }
        String rootDocumentId = DocumentsContract.getTreeDocumentId(treeUri);
        collectDocumentTreeAudioUris(treeUri, rootDocumentId, uris);
        return uris;
    }

    private void collectDocumentTreeAudioUris(Uri treeUri, String documentId, ArrayList<Uri> out) {
        Uri childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, documentId);
        String[] columns = new String[]{
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE
        };
        try (Cursor cursor = getContentResolver().query(childrenUri, columns, null, null, null)) {
            if (cursor == null) {
                return;
            }
            while (cursor.moveToNext()) {
                String childDocumentId = cursor.getString(0);
                String displayName = cursor.getString(1);
                String mimeType = cursor.getString(2);
                if (DocumentsContract.Document.MIME_TYPE_DIR.equals(mimeType)) {
                    collectDocumentTreeAudioUris(treeUri, childDocumentId, out);
                } else if (isAudioName(displayName)) {
                    out.add(DocumentsContract.buildDocumentUriUsingTree(treeUri, childDocumentId));
                }
            }
        }
    }

    private void writePickedAudioUris(String resultName, ArrayList<Uri> uris) throws IOException {
        ArrayList<String> copiedPaths = new ArrayList<>();
        int index = 0;
        for (Uri uri : uris) {
            String displayName = displayNameForUri(uri, "track-" + index + ".mp3");
            if (!isAudioName(displayName)) {
                index += 1;
                continue;
            }
            File copied = copyUriToMediaFile(uri, displayName, index);
            if (copied != null) {
                copiedPaths.add(copied.getAbsolutePath());
            }
            index += 1;
        }
        Collections.sort(copiedPaths);
        writeAtomic(resultName + ".paths", String.join("\n", copiedPaths) + "\n");
    }

    private void writePlaylistImport(Uri uri) throws IOException {
        if (uri == null) {
            writeCancel("playlist_import");
            return;
        }
        String text = readUriText(uri);
        writeAtomic("playlist_import.m3u", text);
    }

    private void writePlaylistExport(Uri uri) throws IOException {
        if (uri == null) {
            writeCancel("playlist_export");
            return;
        }
        try (OutputStream output = getContentResolver().openOutputStream(uri, "wt")) {
            if (output == null) {
                throw new IOException("Android returned no output stream");
            }
            output.write(pendingExportText.getBytes(StandardCharsets.UTF_8));
        }
        writeAtomic("playlist_export.ok", uri.toString());
    }

    private void writeSkinImport(Uri uri) throws IOException {
        if (uri == null) {
            writeCancel("skin_import");
            return;
        }
        String displayName = displayNameForUri(uri, "skin.wsz");
        File copied = copyUriToSkinFile(uri, displayName);
        writeAtomic("skin_import.path", copied.getAbsolutePath());
    }

    private String readUriText(Uri uri) throws IOException {
        try (InputStream input = getContentResolver().openInputStream(uri)) {
            if (input == null) {
                throw new IOException("Android returned no input stream");
            }
            byte[] buffer = new byte[8192];
            StringBuilder text = new StringBuilder();
            int read;
            while ((read = input.read(buffer)) >= 0) {
                text.append(new String(buffer, 0, read, StandardCharsets.UTF_8));
            }
            return text.toString();
        }
    }

    private File copyUriToMediaFile(Uri uri, String displayName, int index) throws IOException {
        File mediaDir = ensureMediaDir();
        String safeName = safeFileName(displayName);
        if (safeName.isEmpty()) {
            safeName = "track-" + index + ".mp3";
        }
        File outputFile = uniqueFile(mediaDir, safeName, index);
        try (
                InputStream input = getContentResolver().openInputStream(uri);
                OutputStream output = new FileOutputStream(outputFile)
        ) {
            if (input == null) {
                throw new IOException("Android returned no input stream");
            }
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                output.write(buffer, 0, read);
            }
        }
        return outputFile;
    }

    private File copyUriToSkinFile(Uri uri, String displayName) throws IOException {
        File skinDir = ensureSkinDir();
        String safeName = safeFileName(displayName);
        if (safeName.isEmpty()) {
            safeName = "skin.wsz";
        }
        if (!safeName.toLowerCase(Locale.US).endsWith(".wsz")
                && !safeName.toLowerCase(Locale.US).endsWith(".zip")) {
            safeName = safeName + ".wsz";
        }
        File outputFile = uniqueFile(skinDir, safeName, 0);
        try (
                InputStream input = getContentResolver().openInputStream(uri);
                OutputStream output = new FileOutputStream(outputFile)
        ) {
            if (input == null) {
                throw new IOException("Android returned no input stream");
            }
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                output.write(buffer, 0, read);
            }
        }
        return outputFile;
    }

    private File uniqueFile(File directory, String safeName, int index) {
        String baseName = safeName;
        String extension = "";
        int dot = safeName.lastIndexOf('.');
        if (dot > 0) {
            baseName = safeName.substring(0, dot);
            extension = safeName.substring(dot);
        }
        File candidate = new File(directory, safeName);
        if (!candidate.exists()) {
            return candidate;
        }
        return new File(directory, baseName + "-" + System.currentTimeMillis() + "-" + index + extension);
    }

    private String displayNameForUri(Uri uri, String fallback) {
        ContentResolver resolver = getContentResolver();
        try (Cursor cursor = resolver.query(uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                String displayName = cursor.getString(0);
                if (displayName != null && !displayName.isEmpty()) {
                    return displayName;
                }
            }
        } catch (Exception ignored) {
        }
        String lastSegment = uri.getLastPathSegment();
        return lastSegment == null || lastSegment.isEmpty() ? fallback : lastSegment;
    }

    private boolean isAudioName(String name) {
        if (name == null) {
            return false;
        }
        int dot = name.lastIndexOf('.');
        if (dot < 0 || dot == name.length() - 1) {
            return false;
        }
        String extension = name.substring(dot + 1).toLowerCase(Locale.US);
        return AUDIO_EXTENSIONS.contains(extension);
    }

    private String safeFileName(String name) {
        return name.replaceAll("[^A-Za-z0-9._ -]", "_").trim();
    }

    private synchronized File ensureBridgeDir() {
        return ensureAppPrivateDir("cranamp_bridge");
    }

    private synchronized File ensureMediaDir() {
        return ensureAppPrivateDir("cranamp_media");
    }

    private synchronized File ensureSkinDir() {
        return ensureAppPrivateDir("cranamp_skins");
    }

    private File ensureAppPrivateDir(String name) {
        File baseDir = getFilesDir();
        if (baseDir == null) {
            throw new IllegalStateException("app files directory is unavailable");
        }
        File dir = new File(baseDir, name);
        if (dir.isDirectory()) {
            return dir;
        }
        if (dir.exists()) {
            throw new IllegalStateException("expected directory but found file: " + dir);
        }
        if (!dir.mkdirs() && !dir.isDirectory()) {
            throw new IllegalStateException("failed to create " + dir);
        }
        return dir;
    }

    private void clearResult(String name) {
        deleteResultFile(name + ".paths");
        deleteResultFile(name + ".m3u");
        deleteResultFile(name + ".path");
        deleteResultFile(name + ".ok");
        deleteResultFile(name + ".cancel");
        deleteResultFile(name + ".error");
    }

    private void writeCancel(String name) {
        if (name.isEmpty()) {
            return;
        }
        writeAtomic(name + ".cancel", "");
    }

    private void writeError(String name, String error) {
        if (name.isEmpty()) {
            return;
        }
        writeAtomic(name + ".error", error == null ? "Android picker failed" : error);
    }

    private void writeAtomic(String fileName, String text) {
        File dir = ensureBridgeDir();
        File tmp = new File(dir, fileName + ".tmp");
        File out = new File(dir, fileName);
        try (FileOutputStream stream = new FileOutputStream(tmp)) {
            stream.write(text.getBytes(StandardCharsets.UTF_8));
        } catch (IOException error) {
            return;
        }
        if (!tmp.renameTo(out)) {
            deleteResultFile(fileName);
            tmp.renameTo(out);
        }
    }

    private void deleteResultFile(String fileName) {
        File file = new File(ensureBridgeDir(), fileName);
        if (file.isFile()) {
            file.delete();
        }
    }

    private String resultNameForRequest(int requestCode) {
        switch (requestCode) {
            case REQ_AUDIO_REPLACE:
            case REQ_FOLDER_REPLACE:
                return "audio_replace";
            case REQ_AUDIO_APPEND:
            case REQ_FOLDER_APPEND:
                return "audio_append";
            case REQ_IMPORT_PLAYLIST:
                return "playlist_import";
            case REQ_EXPORT_PLAYLIST:
                return "playlist_export";
            case REQ_IMPORT_SKIN:
                return "skin_import";
            default:
                return "";
        }
    }
}
