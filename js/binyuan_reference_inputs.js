import { app } from "/scripts/app.js";

const imagePort = /^upstream_image_[1-9][0-9]*$/;

function refreshReferences(node) {
    if (node._bqRefreshingReferences) return;
    node._bqRefreshingReferences = true;
    try {
        // removeInput updates graph link target slots; never splice inputs directly.
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const input = node.inputs[i];
            if (imagePort.test(input.name) && input.link == null) node.removeInput(i);
        }
        const references = node.inputs.filter(input => imagePort.test(input.name));
        const count = Math.max(2, references.length + 1);
        while (references.length < count) {
            node.addInput(`upstream_image_${references.length + 1}`, "IMAGE");
            references.push(node.inputs[node.inputs.length - 1]);
        }
        references.forEach((input, index) => {
            input.name = `upstream_image_${index + 1}`;
            input.label = `Upstream image ${index + 1} ｜ 上游图像 ${index + 1}`;
        });
        node.setDirtyCanvas?.(true, true);
    } finally {
        node._bqRefreshingReferences = false;
    }
}

function scheduleRefresh(node) {
    if (node._bqRefreshingReferences || node._bqReferenceTimer != null) return;
    // Wait until connection replacement or workflow restoration has finished.
    node._bqReferenceTimer = setTimeout(() => {
        node._bqReferenceTimer = null;
        refreshReferences(node);
    }, 0);
}

app.registerExtension({
    name: "BinyuanUltimateSamplerEN.ReferenceInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "BinyuanUltimateSamplerEN") return;
        for (const event of ["onNodeCreated", "onConfigure", "onAdded"]) {
            const previous = nodeType.prototype[event];
            nodeType.prototype[event] = function () {
                const result = previous?.apply(this, arguments);
                scheduleRefresh(this);
                return result;
            };
        }
        const previous = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index) {
            const result = previous?.apply(this, arguments);
            if (type === 1 && imagePort.test(this.inputs[index]?.name)) scheduleRefresh(this);
            return result;
        };
    },
});
