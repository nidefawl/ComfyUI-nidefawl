import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { $el } from "../../../scripts/ui.js";

app.registerExtension({
  name: "Comfy.nidefawl.FlushCache",
  setup() {
    const checkBoxFlushCache = $el("div", {}, [
      $el("label", { innerHTML: "Flush Cache" }, [
        $el("input", {
          type: "checkbox",
          id: "flush-cache",
        }),
      ]),
    ]);
    document.querySelector("button.comfy-queue-btn").after(checkBoxFlushCache);
  },
  
  async beforeConfigureGraph(graphData, missingNodeTypes) {
    const checkBoxFlushCache = document.querySelector("#flush-cache");
    if (checkBoxFlushCache) {
      if (graphData['config'] && graphData['config']['flush_cache']) {
        checkBoxFlushCache.checked = graphData['config']['flush_cache'] === !!graphData['config']['flush_cache'];
      } else {
        checkBoxFlushCache.checked = false;
      }
    }
  }
});

const original_queuePrompt = api.queuePrompt;
async function queuePrompt_with_flush_cache(number, { output, workflow }) {
  if (!workflow.config) {
    workflow.config = {};
  }
  const flushCache = document.querySelector("#flush-cache")?.checked === true;
  workflow.config['flush_cache'] = flushCache;
  return original_queuePrompt.call(api, number, { output, workflow });
}

api.queuePrompt = queuePrompt_with_flush_cache;
