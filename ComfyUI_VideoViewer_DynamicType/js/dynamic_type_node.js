import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfy.DynamicTypeNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DynamicTypeNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.serialize_widgets = true;
                this.addWidget("button", "Update Type", null, () => {
                    this.updateDynamicType();
                });

                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);
                this.updateDynamicType();
            }

            nodeType.prototype.updateDynamicType = async function() {
                const operationWidget = this.widgets.find(w => w.name === "operation");
                const inputValueWidget = this.widgets.find(w => w.name === "input_value");

                if (!operationWidget || !inputValueWidget) return;

                const operation = operationWidget.value;
                const inputValue = inputValueWidget.value;

                try {
                    const response = await api.fetchApi(`/get_dynamic_type?operation=${encodeURIComponent(operation)}&input_value=${encodeURIComponent(inputValue)}`);
                    const data = await response.json();

                    if (data.error) {
                        alert(data.error);
                        return;
                    }

                    // Update the node's output type
                    this.outputs[0].type = data.type;
                    this.outputs[0].name = data.type;  // Optionally update the output name

                    // Force redraw of the node
                    app.graph.setDirtyCanvas(true, true);
                } catch (error) {
                    console.error("Error updating dynamic type:", error);
                    alert("Failed to update type. See console for details.");
                }
            };
        }
    }
});
