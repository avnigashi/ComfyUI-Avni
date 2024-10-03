import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfy.VideoViewer",
    async setup() {
        console.log("Video Viewer Extension loaded");
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VideoViewerNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Add View Video button
                this.addWidget("button", "View Video", null, () => {
                    const videoPath = this.widgets.find(w => w.name === "video_path").value;
                    const videoElement = document.createElement("video");
                    videoElement.controls = true;
                    videoElement.style.maxWidth = "100%";
                    videoElement.src = api.apiURL(`/view_video?video_path=${encodeURIComponent(videoPath)}`);

                    videoElement.onerror = (e) => {
                        console.error("Error loading video:", e);
                        app.ui.dialog.show("Error", `Failed to load video: ${e.message}`);
                    };

                    app.ui.dialog.show(videoElement);
                });

                // Add Upload Video button
                this.addWidget("button", "Upload Video", null, () => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.accept = "video/*";
                    input.onchange = async (e) => {
                        const file = e.target.files[0];
                        if (file) {
                            const formData = new FormData();
                            formData.append("video", file);

                            try {
                                const response = await api.fetchApi("/upload_video", {
                                    method: "POST",
                                    body: formData
                                });

                                if (response.status === 200) {
                                    const result = await response.json();
                                    this.widgets.find(w => w.name === "video_path").value = result.filename;
                                    app.graph.setDirtyCanvas(true);
                                    app.ui.dialog.show("Success", "Video uploaded successfully");
                                } else {
                                    const errorText = await response.text();
                                    console.error("Upload failed:", errorText);
                                    app.ui.dialog.show("Error", `Upload failed: ${errorText}`);
                                }
                            } catch (error) {
                                console.error("Upload error:", error);
                                app.ui.dialog.show("Error", `Upload error: ${error.message}`);
                            }
                        }
                    };
                    input.click();
                });

                return r;
            };
        }
    }
});
