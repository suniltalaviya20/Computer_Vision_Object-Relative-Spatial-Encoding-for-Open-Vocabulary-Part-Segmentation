// ============================================================
// CONSTANTS
// ============================================================

const CATALOGUE_URL =
  "data/catalogue.json";

const INFERENCE_OPTIONS_URL =
  "data/inference_options.json";


const VIEW_GROUPS = [

  {
    label: "RESULT",

    views: [
      {
        id: "prediction",
        label: "Prediction overlay",
      },

      {
        id: "errors",
        label: "Prediction vs GT",
      },

      {
        id: "probability",
        label: "Part probability",
      },
    ],
  },


  {
    label: "IMAGE",

    views: [
      {
        id: "original",
        label: "Original image",
      },

      {
        id: "parent",
        label: "Parent mask",
      },
    ],
  },


  {
    label: "MODEL INPUTS",

    views: [
      {
        id: "x",
        label: "Absolute X",
      },

      {
        id: "y",
        label: "Absolute Y",
      },

      {
        id: "u",
        label: "Relative U",
      },

      {
        id: "v",
        label: "Relative V",
      },

      {
        id: "d",
        label: "Boundary distance D",
      },

      {
        id: "alignment",
        label: "Text-image alignment",
      },
    ],
  },


  {
    label: "GATING",

    views: [
      {
        id: "gate",
        label: "Learned U/V/D weights",
      },
    ],
  },

];



// ============================================================
// STATE
// ============================================================

const state = {

  catalogue: null,

  models: [],

  samples: [],

  robustnessConditions: [],

  sample: null,

  splitId: "test_unseen",

  conditionId: "clean",

  comparing: true,

  viewers: {

    a: {
      modelId: null,
      viewId: "prediction",
    },

    b: {
      modelId: null,
      viewId: "prediction",
    },

  },

};


const uploadState = {

  imageFile: null,

  maskFile: null,

  imageURL: null,

  maskURL: null,

  imageSize: null,

  maskSize: null,

  running: false,

  apiOnline: false,

  apiCheckRunning: false,

  apiRetryTimer: null,

  selectedModel: null,

  modelSelectionChanged: false,

  automaticParentAvailable: false,

  parentCategories: [],

  partQueries: [],

  partSuggestionGroups: {},

  parentPartGroups: {},

  detections: [],

  maskReady: false,

  brushMode: "add",

  drawing: false,

  lastPoint: null,

  initialMask: null,

  undoStack: [],

};



// ============================================================
// DOM HELPERS
// ============================================================

function $(
  selector
) {
  return document.querySelector(
    selector
  );
}


function $$(
  selector
) {
  return Array.from(
    document.querySelectorAll(
      selector
    )
  );
}



// ============================================================
// TEXT HELPERS
// ============================================================

function human(
  value
) {

  if (
    value === null
    || value === undefined
  ) {
    return "—";
  }


  return String(
    value
  )
    .replaceAll(
      "_",
      " "
    )
    .replace(
      /\b\w/g,
      (character) =>
        character.toUpperCase()
    );
}



// ============================================================
// ASSET URL
// ============================================================

function assetURL(
  asset
) {

  if (!asset) {
    return null;
  }


  const file =
    typeof asset === "string"
      ? asset
      : asset.file;


  if (!file) {
    return null;
  }


  return (
    `${file}?v=5`
  );
}



// ============================================================
// CURRENT CONDITION
// ============================================================

function currentCondition() {

  return (
    state.robustnessConditions.find(
      (condition) =>
        condition.id
        === state.conditionId
    )
    || {
      id: "clean",
      label: "Clean",
      type: "clean",
      value: null,
    }
  );
}



// ============================================================
// CONDITION DATA FOR CURRENT SAMPLE
//
// Clean uses the sample's normal assets/results.
//
// Every robustness condition uses:
// sample.robustness[conditionId]
// ============================================================

function conditionData(
  sample = state.sample
) {

  if (!sample) {
    return null;
  }


  if (
    state.conditionId
    === "clean"
  ) {

    return {
      id: "clean",

      label: "Clean",

      type: "clean",

      value: null,

      assets:
        sample.assets
        || {},

      results:
        sample.results
        || {},
    };

  }


  return (
    sample.robustness?.[
      state.conditionId
    ]
    || null
  );
}



// ============================================================
// IS CONDITION AVAILABLE?
// ============================================================

function sampleHasCondition(
  sample,
  conditionId
) {

  if (!sample) {
    return false;
  }


  if (
    conditionId
    === "clean"
  ) {
    return true;
  }


  return Boolean(
    sample.robustness?.[
      conditionId
    ]
  );
}



// ============================================================
// GET RESULT FOR MODEL
// ============================================================

function resultFor(
  modelId
) {

  const data =
    conditionData();


  if (!data) {
    return null;
  }


  return (
    data.results?.[
      modelId
    ]
    || null
  );
}



// ============================================================
// GET CONDITION-SPECIFIC COMMON ASSET
// ============================================================

function commonAsset(
  viewId
) {

  const data =
    conditionData();


  if (!data) {
    return null;
  }


  return (
    data.assets?.[
      viewId
    ]
    || null
  );
}



// ============================================================
// MODEL LOOKUP
// ============================================================

function modelFor(
  modelId
) {

  return (
    state.models.find(
      (model) =>
        model.id === modelId
    )
    || null
  );
}



// ============================================================
// ASSET FOR A VIEW
// ============================================================

function assetFor(
  modelId,
  viewId
) {

  if (
    !state.sample
  ) {
    return null;
  }


  // --------------------------------------------------------
  // Common condition assets
  // --------------------------------------------------------

  if (
    viewId === "original"
    || viewId === "parent"
  ) {

    return commonAsset(
      viewId
    );

  }


  // --------------------------------------------------------
  // Model-specific assets
  // --------------------------------------------------------

  const result =
    resultFor(
      modelId
    );


  if (!result) {
    return null;
  }


  return (
    result.assets?.[
      viewId
    ]
    || null
  );
}



// ============================================================
// VIEW EXISTS?
// ============================================================

function hasView(
  modelId,
  viewId
) {

  if (!state.sample) {
    return false;
  }


  if (
    viewId
    === "gate"
  ) {

    const model =
      modelFor(
        modelId
      );

    const result =
      resultFor(
        modelId
      );


    return Boolean(
      model?.gated
      && result?.gate_weights
    );

  }


  return Boolean(
    assetFor(
      modelId,
      viewId
    )
  );
}



// ============================================================
// EVALUATION SPLITS
// ============================================================

function splitName(
  splitId
) {

  return splitId === "test_unseen"
    ? "Unseen"
    : "Seen";

}


const UNSEEN_SHOWCASE_MIN_IOU = 0.5;


function catalogueSamplesForSplit(
  splitId
) {

  return state.samples.filter(
    (sample) =>
      sample.split === splitId
  );

}


function bestSampleIoU(
  sample
) {

  const values = Object.values(
    sample.results || {}
  )
    .map(
      (result) =>
        result?.metrics?.iou
    )
    .filter(Number.isFinite);

  return values.length > 0
    ? Math.max(...values)
    : null;

}


function isShowcaseSample(
  sample
) {

  if (
    sample.split !== "test_unseen"
  ) {
    return true;
  }

  const bestIoU =
    bestSampleIoU(sample);

  return bestIoU !== null
    && bestIoU >= UNSEEN_SHOWCASE_MIN_IOU;

}


function samplesForSelectedSplit() {

  return catalogueSamplesForSplit(
    state.splitId
  ).filter(isShowcaseSample);

}


function populateSplitSelect() {

  const select =
    $("#split-select");

  const splits = [
    "test_seen",
    "test_unseen",
  ];

  select.innerHTML = "";

  for (const splitId of splits) {

    const splitSamples =
      catalogueSamplesForSplit(splitId);

    const visibleCount =
      splitSamples.filter(isShowcaseSample).length;

    const option = document.createElement("option");
    option.value = splitId;
    option.textContent =
      `${splitName(splitId)} examples (${visibleCount})`;
    option.disabled = visibleCount === 0;
    select.appendChild(option);

  }

  select.value = state.splitId;

}



// ============================================================
// POPULATE OBJECT SELECT
// ============================================================

function populateObjects() {

  const select =
    $("#object-select");


  const objects = [
    ...new Set(
      samplesForSelectedSplit().map(
        (sample) =>
          sample.object
      )
    ),
  ].sort();


  select.innerHTML = "";


  for (
    const objectName
    of objects
  ) {

    const option =
      document.createElement(
        "option"
      );


    option.value =
      objectName;

    option.textContent =
      human(
        objectName
      );


    select.appendChild(
      option
    );

  }


  select.disabled =
    objects.length === 0;

}



// ============================================================
// POPULATE PART SELECT
// ============================================================

function populateParts(
  preferredPart = null
) {

  const objectSelect =
    $("#object-select");

  const partSelect =
    $("#part-select");


  const objectName =
    objectSelect.value;


  const parts = [
    ...new Set(

      samplesForSelectedSplit()

        .filter(
          (sample) =>
            sample.object
            === objectName
        )

        .map(
          (sample) =>
            sample.part
        )

    ),
  ].sort();


  partSelect.innerHTML =
    "";


  for (
    const partName
    of parts
  ) {

    const option =
      document.createElement(
        "option"
      );


    option.value =
      partName;

    option.textContent =
      human(
        partName
      );


    partSelect.appendChild(
      option
    );

  }


  partSelect.disabled =
    parts.length === 0;


  if (
    preferredPart
    && parts.includes(
      preferredPart
    )
  ) {

    partSelect.value =
      preferredPart;

  }

}



// ============================================================
// POPULATE CONDITION SELECT
// ============================================================

function populateConditions() {

  const select =
    $("#condition-select");


  select.innerHTML =
    "";


  const conditions =
    state.robustnessConditions.length
      ? state.robustnessConditions
      : [
          {
            id: "clean",
            label: "Clean",
          },
        ];


  for (
    const condition
    of conditions
  ) {

    const option =
      document.createElement(
        "option"
      );


    option.value =
      condition.id;

    option.textContent =
      condition.label;


    // ------------------------------------------------------
    // Disable condition if the selected sample does not
    // currently contain exported data for it.
    // ------------------------------------------------------

    option.disabled =
      !sampleHasCondition(
        state.sample,
        condition.id
      );


    select.appendChild(
      option
    );

  }


  if (
    sampleHasCondition(
      state.sample,
      state.conditionId
    )
  ) {

    select.value =
      state.conditionId;

  }

  else {

    state.conditionId =
      "clean";

    select.value =
      "clean";

  }


  select.disabled =
    !state.sample;

}



// ============================================================
// SAMPLES FOR CURRENT OBJECT/PART
// ============================================================

function matchingSamples() {

  const objectName =
    $("#object-select").value;

  const partName =
    $("#part-select").value;


  return state.samples.filter(
    (sample) =>
      sample.split === state.splitId
      && sample.object === objectName
      && sample.part === partName
  );
}



// ============================================================
// SELECT FIRST MATCHING SAMPLE
// ============================================================

function selectFirstMatchingSample() {

  const matches =
    matchingSamples();


  state.sample =
    matches[0]
    || null;


  normalizeCondition();

  render();

}



// ============================================================
// ANOTHER EXAMPLE
// ============================================================

function chooseAnotherExample() {

  const matches =
    matchingSamples();


  if (
    matches.length <= 1
  ) {
    return;
  }


  const currentIndex =
    matches.findIndex(
      (sample) =>
        sample.id
        === state.sample?.id
    );


  const nextIndex =
    (
      currentIndex + 1
    )
    % matches.length;


  state.sample =
    matches[
      nextIndex
    ];


  normalizeCondition();

  render();

  showToast(
    "Loaded another example"
  );

}



// ============================================================
// NORMALIZE CONDITION
// ============================================================

function normalizeCondition() {

  if (
    !sampleHasCondition(
      state.sample,
      state.conditionId
    )
  ) {

    state.conditionId =
      "clean";

  }

}



// ============================================================
// POPULATE MODEL SELECTS
// ============================================================

function populateModels() {

  const selects =
    $$(".model-select");


  for (
    const select
    of selects
  ) {

    select.innerHTML =
      "";


    for (
      const model
      of state.models
    ) {

      const option =
        document.createElement(
          "option"
        );


      option.value =
        model.id;

      option.textContent =
        model.label;


      select.appendChild(
        option
      );

    }

  }


  if (
    state.models.length
  ) {

    state.viewers.a.modelId =
      state.models[0].id;


    state.viewers.b.modelId =
      state.models[
        Math.min(
          1,
          state.models.length - 1
        )
      ].id;

  }

}



// ============================================================
// POPULATE VIEW SELECT FOR ONE VIEWER
// ============================================================

function populateViews(
  viewerId
) {

  const viewer =
    state.viewers[
      viewerId
    ];


  const select =
    $(
      `.view-select[data-viewer="${viewerId}"]`
    );


  select.innerHTML =
    "";


  const availableViews =
    [];


  for (
    const group
    of VIEW_GROUPS
  ) {

    const validViews =
      group.views.filter(
        (view) =>
          hasView(
            viewer.modelId,
            view.id
          )
      );


    if (
      validViews.length === 0
    ) {
      continue;
    }


    const optgroup =
      document.createElement(
        "optgroup"
      );


    optgroup.label =
      group.label;


    for (
      const view
      of validViews
    ) {

      const option =
        document.createElement(
          "option"
        );


      option.value =
        view.id;

      option.textContent =
        view.label;


      optgroup.appendChild(
        option
      );


      availableViews.push(
        view.id
      );

    }


    select.appendChild(
      optgroup
    );

  }


  if (
    !availableViews.includes(
      viewer.viewId
    )
  ) {

    viewer.viewId =
      availableViews.includes(
        "prediction"
      )
        ? "prediction"
        : (
            availableViews[0]
            || null
          );

  }


  if (
    viewer.viewId
  ) {

    select.value =
      viewer.viewId;

  }


  select.disabled =
    availableViews.length === 0;

}



// ============================================================
// VIEW LABEL
// ============================================================

function viewLabel(
  viewId
) {

  for (
    const group
    of VIEW_GROUPS
  ) {

    const view =
      group.views.find(
        (item) =>
          item.id === viewId
      );


    if (view) {
      return view.label;
    }

  }


  return human(
    viewId
  );
}



// ============================================================
// SPACE LABEL
// ============================================================

function spaceLabel(
  asset
) {

  const space =
    asset?.space;


  if (
    space === "crop"
  ) {
    return "model crop";
  }


  if (
    space === "full_image"
  ) {
    return "full image";
  }


  return space
    ? human(
        space
      )
    : "—";
}



// ============================================================
// GATE PANEL
// ============================================================

function renderGate(
  viewerId,
  result
) {

  const stage =
    $(
      `#viewer-${viewerId} .image-stage`
    );


  const image =
    $(
      `[data-image="${viewerId}"]`
    );


  const empty =
    $(
      `[data-empty="${viewerId}"]`
    );


  image.hidden =
    true;

  empty.hidden =
    true;


  let gatePanel =
    stage.querySelector(
      ".gate-panel"
    );


  if (!gatePanel) {

    gatePanel =
      document.createElement(
        "div"
      );

    gatePanel.className =
      "gate-panel";

    stage.appendChild(
      gatePanel
    );

  }


  const weights =
    result?.gate_weights;


  if (!weights) {

    gatePanel.innerHTML =
      `
        <h2>Gate weights unavailable</h2>
        <p>
          This model did not return geometry-gate weights
          for the selected condition.
        </p>
      `;

    return;

  }


  const entries = [
    [
      "U",
      Number(
        weights.u
      ),
    ],

    [
      "V",
      Number(
        weights.v
      ),
    ],

    [
      "D",
      Number(
        weights.d
      ),
    ],
  ];


  gatePanel.innerHTML =
    `
      <h2>Query-conditioned geometry weights</h2>

      ${entries.map(
        ([name, value]) => {

          const percentage =
            Math.max(
              0,
              Math.min(
                100,
                value * 100
              )
            );

          return `
            <div class="gate-row">
              <span>${name}</span>

              <div class="gate-track">
                <div
                  class="gate-fill"
                  style="width:${percentage}%"
                ></div>
              </div>

              <strong class="gate-value">
                ${value.toFixed(3)}
              </strong>
            </div>
          `;

        }
      ).join("")}

      <p>
        These are learned model outputs for the selected
        text query. They should not be interpreted as
        physical orientation.
      </p>
    `;

}



// ============================================================
// REMOVE GATE PANEL
// ============================================================

function removeGatePanel(
  viewerId
) {

  const stage =
    $(
      `#viewer-${viewerId} .image-stage`
    );


  const gatePanel =
    stage.querySelector(
      ".gate-panel"
    );


  if (
    gatePanel
  ) {
    gatePanel.remove();
  }

}



// ============================================================
// RENDER ONE VIEWER
// ============================================================

function renderViewer(
  viewerId
) {

  const viewer =
    state.viewers[
      viewerId
    ];


  const model =
    modelFor(
      viewer.modelId
    );


  const result =
    resultFor(
      viewer.modelId
    );

  renderSplitPerformance(
    viewerId,
    viewer.modelId
  );


  const modelSelect =
    $(
      `.model-select[data-viewer="${viewerId}"]`
    );


  const viewSelect =
    $(
      `.view-select[data-viewer="${viewerId}"]`
    );


  const image =
    $(
      `[data-image="${viewerId}"]`
    );


  const empty =
    $(
      `[data-empty="${viewerId}"]`
    );


  const description =
    $(
      `[data-description="${viewerId}"]`
    );


  const caption =
    $(
      `[data-caption="${viewerId}"]`
    );


  const space =
    $(
      `[data-space="${viewerId}"]`
    );


  modelSelect.disabled =
    !state.sample
    || state.models.length === 0;


  if (
    viewer.modelId
  ) {
    modelSelect.value =
      viewer.modelId;
  }


  populateViews(
    viewerId
  );


  if (
    viewer.viewId
  ) {
    viewSelect.value =
      viewer.viewId;
  }


  description.textContent =
    model?.description
    || "—";


  // --------------------------------------------------------
  // No sample / result
  // --------------------------------------------------------

  if (
    !state.sample
    || !result
  ) {

    removeGatePanel(
      viewerId
    );

    image.hidden =
      true;

    empty.hidden =
      false;

    caption.textContent =
      "No result available";

    space.textContent =
      "—";

    renderMetrics(
      viewerId,
      null
    );

    return;

  }


  // --------------------------------------------------------
  // Gate visualization
  // --------------------------------------------------------

  if (
    viewer.viewId
    === "gate"
  ) {

    renderGate(
      viewerId,
      result
    );

    caption.textContent =
      "Learned geometry weights";

    space.textContent =
      "query-conditioned";

    renderMetrics(
      viewerId,
      result
    );

    return;

  }


  removeGatePanel(
    viewerId
  );


  // --------------------------------------------------------
  // Image-based view
  // --------------------------------------------------------

  const asset =
    assetFor(
      viewer.modelId,
      viewer.viewId
    );


  const url =
    assetURL(
      asset
    );


  if (!url) {

    image.hidden =
      true;

    empty.hidden =
      false;

    caption.textContent =
      "View unavailable";

    space.textContent =
      "—";

  }

  else {

    empty.hidden =
      true;

    image.hidden =
      false;

    image.src =
      url;

    caption.textContent =
      asset.description
      || viewLabel(
          viewer.viewId
        );

    space.textContent =
      spaceLabel(
        asset
      );

  }


  renderMetrics(
    viewerId,
    result
  );

}



// ============================================================
// METRICS
// ============================================================

function renderMetrics(
  viewerId,
  result
) {

  const iou =
    $(
      `[data-iou="${viewerId}"]`
    );

  const dice =
    $(
      `[data-dice="${viewerId}"]`
    );

  const leakage =
    $(
      `[data-leakage="${viewerId}"]`
    );

  if (!result) {

    iou.textContent =
      "—";

    dice.textContent =
      "—";

    leakage.textContent =
      "—";

    return;

  }


  const metrics =
    result.metrics
    || {};


  iou.textContent =
    formatMetric(
      metrics.iou
    );

  dice.textContent =
    formatMetric(
      metrics.dice
    );

  leakage.textContent =
    formatMetric(
      metrics.leakage
    );


}



// ============================================================
// SPLIT PERFORMANCE
// ============================================================

function meanMetricForSplit(
  modelId,
  metricName
) {

  const values = catalogueSamplesForSplit(state.splitId)
    .map(
      (sample) =>
        sample.results?.[modelId]?.metrics?.[metricName]
    )
    .filter(Number.isFinite);

  if (values.length === 0) {
    return null;
  }

  return values.reduce(
    (total, value) => total + value,
    0
  ) / values.length;

}


function renderSplitPerformance(
  viewerId,
  modelId
) {

  const label = $(`[data-split-label="${viewerId}"]`);
  const average = $(`[data-split-iou="${viewerId}"]`);
  const delta = $(`[data-split-delta="${viewerId}"]`);
  const splitSamples = catalogueSamplesForSplit(state.splitId);
  const mean = meanMetricForSplit(modelId, "iou");
  const baselineId = "final_baseline_object_mask";
  const baselineMean = meanMetricForSplit(baselineId, "iou");

  label.textContent =
    `${splitName(state.splitId)} catalogue · ${splitSamples.length} examples`;

  average.textContent =
    mean === null
      ? "Mean IoU —"
      : `Mean IoU ${formatMetric(mean)}`;

  delta.classList.remove("is-positive", "is-negative");

  if (mean === null || baselineMean === null) {
    delta.textContent = "No aggregate available";
    return;
  }

  if (modelId === baselineId) {
    delta.textContent = "Baseline reference";
    return;
  }

  const difference = (mean - baselineMean) * 100;
  delta.textContent =
    `${difference >= 0 ? "+" : ""}${difference.toFixed(1)} pp vs baseline`;
  delta.classList.add(
    difference >= 0
      ? "is-positive"
      : "is-negative"
  );

}



// ============================================================
// METRIC FORMAT
// ============================================================

function formatMetric(
  value
) {

  if (
    value === undefined
    || value === null
    || Number.isNaN(
      Number(
        value
      )
    )
  ) {
    return "—";
  }


  return Number(
    value
  ).toFixed(
    3
  );
}



// ============================================================
// GROUND TRUTH
// ============================================================

function renderGroundTruthPreview() {

  const image =
    $("#gt-preview-image");

  const empty =
    $("#gt-preview-empty");

  const label =
    $("#gt-preview-label");

  const conditionLabel =
    $("#gt-condition-label");


  if (!state.sample) {

    label.textContent =
      "No part selected";

    conditionLabel.textContent =
      "Dataset annotation";

    image.hidden =
      true;

    empty.hidden =
      false;

    return;

  }


  label.textContent =
    `${human(
      state.sample.object
    )} / ${human(
      state.sample.part
    )}`;


  const condition =
    currentCondition();


  conditionLabel.textContent =
    condition.id === "clean"
      ? "Dataset annotation"
      : condition.label;


  const data =
    conditionData();


  const asset =
    data?.assets?.ground_truth_overlay;


  const url =
    assetURL(
      asset
    );


  if (!url) {

    image.hidden =
      true;

    empty.hidden =
      false;

    empty.textContent =
      "Ground truth unavailable for this condition";

    return;

  }


  empty.hidden =
    true;

  image.hidden =
    false;

  image.src =
    url;

}



// ============================================================
// EXAMPLE SUMMARY
// ============================================================

function renderExampleSummary() {

  const name =
    $("#example-name");

  const splitLabel =
    $("#example-split-label");

  if (!state.sample) {

    name.textContent =
      "—";

    splitLabel.textContent =
      "CURRENT EXAMPLE";

    return;

  }


  name.textContent =
    `${human(
      state.sample.object
    )} / ${human(
      state.sample.part
    )}`;

  splitLabel.textContent =
    `${splitName(state.sample.split).toUpperCase()} EXAMPLE`;
}



// ============================================================
// CONDITION LABEL
// ============================================================

function renderConditionLabel() {

  const condition =
    currentCondition();


  $("#viewer-condition-tag")
    .textContent =
      condition.label;

}



// ============================================================
// RANDOM BUTTON
// ============================================================

function renderRandomButton() {

  const button =
    $("#random-button");


  const matches =
    matchingSamples();


  button.disabled =
    matches.length <= 1;

}



// ============================================================
// COMPARISON STATE
// ============================================================

function renderComparison() {

  const grid =
    $("#viewer-grid");

  const viewerB =
    $("#viewer-b");

  const button =
    $("#compare-button");

  const summary =
    $("#comparison-summary");


  grid.classList.toggle(
    "comparing",
    state.comparing
  );


  viewerB.hidden =
    !state.comparing;


  button.setAttribute(
    "aria-pressed",
    state.comparing
      ? "true"
      : "false"
  );


  button.textContent =
    state.comparing
      ? "Exit comparison"
      : "Compare models";


  summary.hidden =
    !state.comparing;

}



// ============================================================
// RENDER EVERYTHING
// ============================================================

function render() {

  populateConditions();

  renderExampleSummary();

  renderGroundTruthPreview();

  renderConditionLabel();

  renderRandomButton();

  renderComparison();

  renderViewer(
    "a"
  );


  if (
    state.comparing
  ) {

    renderViewer(
      "b"
    );

  }

  refreshEnhancedUI();

}



// ============================================================
// SPLIT CHANGE
// ============================================================

function handleSplitChange(
  event
) {

  state.splitId =
    event.target.value;

  populateObjects();
  populateParts();
  selectFirstMatchingSample();

}



// ============================================================
// OBJECT CHANGE
// ============================================================

function handleObjectChange() {

  populateParts();

  selectFirstMatchingSample();

}



// ============================================================
// PART CHANGE
// ============================================================

function handlePartChange() {

  selectFirstMatchingSample();

}



// ============================================================
// CONDITION CHANGE
// ============================================================

function handleConditionChange(
  event
) {

  state.conditionId =
    event.target.value;


  // --------------------------------------------------------
  // If views are linked, keep both on the same view whenever
  // that view exists under the new condition.
  // --------------------------------------------------------

  if (
    state.comparing
  ) {

    state.viewers.b.viewId =
      state.viewers.a.viewId;

  }


  render();

}



// ============================================================
// MODEL CHANGE
// ============================================================

function handleModelChange(
  event
) {

  const viewerId =
    event.target.dataset.viewer;


  state.viewers[
    viewerId
  ].modelId =
    event.target.value;


  render();

}



// ============================================================
// VIEW CHANGE
// ============================================================

function handleViewChange(
  event
) {

  const viewerId =
    event.target.dataset.viewer;


  const viewId =
    event.target.value;


  state.viewers[
    viewerId
  ].viewId =
    viewId;


  if (
    state.comparing
  ) {

    const otherViewer =
      viewerId === "a"
        ? "b"
        : "a";


    if (
      hasView(
        state.viewers[
          otherViewer
        ].modelId,
        viewId
      )
    ) {

      state.viewers[
        otherViewer
      ].viewId =
        viewId;

    }

  }


  render();

}



// ============================================================
// COMPARE
// ============================================================

function toggleCompare() {

  state.comparing =
    !state.comparing;


  if (
    state.comparing
  ) {

    state.viewers.b.viewId =
      state.viewers.a.viewId;

  }


  render();

  showToast(
    state.comparing
      ? "Comparison mode enabled"
      : "Comparison mode closed"
  );

}



// ============================================================
// INITIAL SAMPLE
// ============================================================

function chooseInitialSample() {

  const splitSamples =
    samplesForSelectedSplit();

  if (
    splitSamples.length === 0
  ) {
    return;
  }


  const preferredSample = splitSamples.find(
    (sample) =>
      sample.id === "val:2008_007814:part_105"
  );


  state.sample =
    preferredSample
    || splitSamples[0];


  const objectSelect =
    $("#object-select");


  objectSelect.value =
    state.sample.object;


  populateParts(
    state.sample.part
  );


  $("#part-select").value =
    state.sample.part;

}



// ============================================================
// VALIDATE CATALOGUE
// ============================================================

function validateCatalogue(
  catalogue
) {

  if (
    !catalogue
    || !Array.isArray(
      catalogue.models
    )
    || !Array.isArray(
      catalogue.samples
    )
  ) {

    throw new Error(
      "Invalid catalogue.json"
    );

  }

}



// ============================================================
// DEFAULT ROBUSTNESS CONDITIONS
//
// These allow the site to continue working even before the
// robustness exporter has been run.
// ============================================================

function defaultConditions() {

  return [
    {
      id: "clean",
      label: "Clean",
      type: "clean",
      value: null,
    },
  ];

}



// ============================================================
// LOAD CATALOGUE
// ============================================================

async function loadCatalogue() {

  try {

    const response =
      await fetch(
        CATALOGUE_URL,
        {
          cache: "no-store",
        }
      );


    if (
      !response.ok
    ) {

      throw new Error(
        `HTTP ${response.status}`
      );

    }


    const catalogue =
      await response.json();


    validateCatalogue(
      catalogue
    );


    state.catalogue =
      catalogue;

    state.models =
      catalogue.models;

    state.samples =
      catalogue.samples;

    state.robustnessConditions =
      Array.isArray(
        catalogue.robustness_conditions
      )
        ? catalogue.robustness_conditions
        : defaultConditions();


    // ------------------------------------------------------
    // Guarantee Clean exists and appears first.
    // ------------------------------------------------------

    if (
      !state.robustnessConditions.some(
        (condition) =>
          condition.id === "clean"
      )
    ) {

      state.robustnessConditions.unshift(
        {
          id: "clean",
          label: "Clean",
          type: "clean",
          value: null,
        }
      );

    }


    state.robustnessConditions.sort(
      (
        a,
        b
      ) => {

        if (
          a.id === "clean"
        ) {
          return -1;
        }

        if (
          b.id === "clean"
        ) {
          return 1;
        }

        return 0;

      }
    );


    populateSplitSelect();

    populateObjects();

    populateModels();

    chooseInitialSample();

    populateUploadControls();
    render();

    document.body.classList.remove(
      "is-loading"
    );

    showToast(
      "Research demo ready"
    );

  }

  catch (
    error
  ) {

    console.error(
      error
    );
    $("#setup-notice")
      .hidden =
        false;


    document.body.classList.remove(
      "is-loading"
    );

  }

}



// ============================================================
// EVENTS
// ============================================================

$("#split-select")
  .addEventListener(
    "change",
    handleSplitChange
  );


$("#object-select")
  .addEventListener(
    "change",
    handleObjectChange
  );


$("#part-select")
  .addEventListener(
    "change",
    handlePartChange
  );


$("#condition-select")
  .addEventListener(
    "change",
    handleConditionChange
  );


$("#random-button")
  .addEventListener(
    "click",
    chooseAnotherExample
  );


$("#compare-button")
  .addEventListener(
    "click",
    toggleCompare
  );


for (
  const select
  of $$(".model-select")
) {

  select.addEventListener(
    "change",
    handleModelChange
  );

}


for (
  const select
  of $$(".view-select")
) {

  select.addEventListener(
    "change",
    handleViewChange
  );

}



// ============================================================
// START
// ============================================================

let toastTimer = null;


function setupEnhancedExperience() {

  document.body.classList.add("is-loading");
  const toast = document.createElement("div");

  toast.id = "demo-toast";
  toast.className = "demo-toast";
  toast.setAttribute("role", "status");
  toast.setAttribute("aria-live", "polite");
  document.body.appendChild(toast);

  const lightbox = document.createElement("div");

  lightbox.id = "image-lightbox";
  lightbox.className = "image-lightbox";
  lightbox.hidden = true;
  lightbox.innerHTML = `
    <button
      class="lightbox-close"
      type="button"
      aria-label="Close expanded image"
    >×</button>
    <div class="lightbox-content">
      <div class="lightbox-gallery"></div>
      <p></p>
    </div>
  `;

  document.body.appendChild(lightbox);

  lightbox.addEventListener("click", (event) => {

    if (
      event.target === lightbox
      || event.target.closest(".lightbox-close")
    ) {
      closeLightbox();
    }

  });

  document.addEventListener("click", handleEnhancedClick);
  document.addEventListener("keydown", handleKeyboardShortcuts);

  addMetricHelp();
  addZoomControls();

}


function refreshEnhancedUI() {

  const values = {
    samples: state.samples.length,
    models: state.models.length,
    conditions: state.robustnessConditions.length,
  };

  for (const [name, value] of Object.entries(values)) {

    const element = $(
      `[data-stat="${name}"]`
    );

    if (element) {
      element.textContent = value || "—";
    }

  }

  document.title = state.sample
    ? `${human(state.sample.object)} / ${human(state.sample.part)} · Part Segmentation`
    : "Object-Relative Part Segmentation";

  for (const button of $$(".zoom-button")) {

    const stage = button.closest(
      ".image-stage, .ground-truth-stage"
    );

    const image = stage?.querySelector(
      ".stage-image:not([hidden]), .gt-preview-image:not([hidden])"
    );

    button.hidden = !image;

  }

}


function addMetricHelp() {

  const help = [
    ["[data-iou]", "Intersection over Union: higher is better."],
    ["[data-dice]", "Dice overlap score: higher is better."],
    ["[data-leakage]", "Prediction outside the parent object: lower is better."],
  ];

  for (const [selector, message] of help) {

    for (const value of $$(selector)) {

      const metric = value.closest(".metric");

      if (metric) {
        metric.title = message;
      }

    }

  }

}


function addZoomControls() {

  const stages = [
    $(".ground-truth-stage"),
    ...$$(".image-stage"),
  ].filter(Boolean);

  for (const stage of stages) {

    if (stage.querySelector(".zoom-button")) {
      continue;
    }

    const button = document.createElement("button");

    button.className = "zoom-button";
    button.type = "button";
    button.setAttribute("aria-label", "Expand image");
    button.title = "Expand image";
    button.innerHTML = `
      <span aria-hidden="true">⛶</span>
      <span>Expand</span>
    `;

    button.addEventListener("click", (event) => {

      event.stopPropagation();

      const image = stage.querySelector(
        ".stage-image:not([hidden]), .gt-preview-image:not([hidden])"
      );

      if (image && image.src) {
        openLightbox(image);
      }
      else {
        showToast("No image available to expand");
      }

    });

    stage.appendChild(button);

  }

}


function handleEnhancedClick(event) {

  if (!(event.target instanceof Element)) {
    return;
  }

  const image = event.target.closest(
    ".stage-image, .gt-preview-image"
  );

  if (image && !image.hidden && image.src) {
    openLightbox(image);
    return;
  }


}


function handleKeyboardShortcuts(event) {

  if (!(event.target instanceof Element)) {
    return;
  }

  const tag = event.target.tagName;

  if (
    event.target.isContentEditable
    || ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(tag)
  ) {
    return;
  }

  if (event.key === "Escape") {
    closeLightbox();
    return;
  }

  if (event.key.toLowerCase() === "c") {
    toggleCompare();
  }

  if (
    event.key.toLowerCase() === "r"
    && !$("#random-button").disabled
  ) {
    chooseAnotherExample();
  }

}


function openLightbox(sourceImage) {

  const lightbox = $("#image-lightbox");
  const gallery = lightbox.querySelector(".lightbox-gallery");
  const caption = lightbox.querySelector("p");
  const isViewerImage = Boolean(
    sourceImage.closest(".viewer-card")
  );

  let images = [sourceImage];

  if (
    state.comparing
    && isViewerImage
  ) {

    images = [
      $("#viewer-a"),
      $("#viewer-b"),
    ]
      .map((viewer) => viewer?.querySelector(
        ".stage-image:not([hidden])"
      ))
      .filter((image) => image?.src);

  }

  gallery.innerHTML = "";
  gallery.classList.toggle(
    "is-comparing",
    images.length > 1
  );

  for (const image of images) {

    const figure = document.createElement("figure");
    const expandedImage = document.createElement("img");
    const label = document.createElement("figcaption");
    const labelTitle = document.createElement("strong");
    const labelDetail = document.createElement("span");
    const viewer = image.closest(".viewer-card");
    const modelSelect = viewer?.querySelector(".model-select");
    const selectedModel = modelSelect?.selectedOptions?.[0]?.textContent;
    const viewCaption = viewer
      ?.querySelector(".image-caption span")
      ?.textContent;

    expandedImage.src = image.src;
    expandedImage.alt = image.alt;
    labelTitle.textContent = selectedModel || "Ground truth";
    labelDetail.textContent = selectedModel
      ? viewCaption || "Model visualization"
      : "Dataset reference";

    label.appendChild(labelTitle);
    label.appendChild(labelDetail);
    figure.appendChild(expandedImage);
    figure.appendChild(label);
    gallery.appendChild(figure);

  }

  caption.textContent = images.length > 1
    ? "Side-by-side model comparison · both panels use the same sample and condition"
    : sourceImage.closest(".ground-truth-section")
      ? "Ground-truth reference"
      : sourceImage.closest(".viewer-card")
          ?.querySelector(".image-caption span")
          ?.textContent
        || "Model visualization";

  lightbox.hidden = false;
  document.body.classList.add("lightbox-open");
  lightbox.querySelector(".lightbox-close").focus();

}


function closeLightbox() {

  const lightbox = $("#image-lightbox");

  if (!lightbox || lightbox.hidden) {
    return;
  }

  lightbox.hidden = true;
  document.body.classList.remove("lightbox-open");

}

function showToast(message) {

  const toast = $("#demo-toast");

  if (!toast) {
    return;
  }

  window.clearTimeout(toastTimer);
  toast.textContent = message;
  toast.classList.add("is-visible");

  toastTimer = window.setTimeout(() => {
    toast.classList.remove("is-visible");
  }, 2200);

}


// ============================================================
// USER IMAGE INFERENCE
// ============================================================

const API_BASE_URL = (
  document.querySelector(
    'meta[name="part-segmentation-api"]'
  )?.content || ""
).trim().replace(/\/$/, "");


function inferenceURL(path) {

  return `${API_BASE_URL}${path}`;

}


function uploadCategories() {

  const configured = state.catalogue?.object_parts;

  if (configured && typeof configured === "object") {

    return Object.entries(configured)
      .map(([object, parts]) => ({
        object,
        parts: [...parts],
      }))
      .sort((a, b) => a.object.localeCompare(b.object));

  }

  const partsByObject = new Map();

  for (const sample of state.samples) {

    if (!partsByObject.has(sample.object)) {
      partsByObject.set(sample.object, new Set());
    }

    partsByObject.get(sample.object).add(sample.part);

  }

  return [...partsByObject.entries()]
    .map(([object, parts]) => ({
      object,
      parts: [...parts].sort(),
    }))
    .sort((a, b) => a.object.localeCompare(b.object));

}


function currentUploadObject() {

  return $("#upload-object-input").value.trim();

}


function currentUploadPart() {

  return $("#upload-part-input").value.trim();

}


function currentUploadCategory() {

  const objectName = currentUploadObject().toLowerCase();
  return uploadCategories().find(
    (category) => category.object.toLowerCase() === objectName
  );

}


function partSuggestionsForCurrentObject() {

  const objectName = currentUploadObject().toLowerCase();

  if (!uploadState.parentCategories.includes(objectName)) {
    return [];
  }

  const category = currentUploadCategory();
  if (category) {
    return category.parts;
  }

  const groupName = uploadState.parentPartGroups[objectName];
  return uploadState.partSuggestionGroups[groupName] || [];

}


function updateExperimentalNote() {

  const category = currentUploadCategory();
  const objectName = currentUploadObject().toLowerCase();
  const partName = currentUploadPart().toLowerCase();
  const supportedObject = uploadState.parentCategories.includes(objectName);
  const evaluatedPair = Boolean(
    category && category.parts.some((part) => part.toLowerCase() === partName)
  );

  $("#experimental-query-note").hidden = !(
    supportedObject && partName && !evaluatedPair
  );

}


function populateUploadSuggestions() {

  for (const menu of $$(".autocomplete-menu")) {
    menu.innerHTML = "";
    menu.hidden = true;
  }

}


function autocompleteMatches(values, query) {

  const normalised = query.trim().toLowerCase();

  if (!normalised) {
    return [...values];
  }

  return [...values]
    .filter((value) => value.toLowerCase().includes(normalised))
    .sort((a, b) => {
      const aStarts = a.toLowerCase().startsWith(normalised) ? 0 : 1;
      const bStarts = b.toLowerCase().startsWith(normalised) ? 0 : 1;
      return aStarts - bStarts || a.localeCompare(b);
    });

}


function setupAutocomplete(
  inputSelector,
  menuSelector,
  valuesProvider,
  onChoose = () => {}
) {

  const input = $(inputSelector);
  const menu = $(menuSelector);
  let activeIndex = -1;

  function close() {
    menu.hidden = true;
    activeIndex = -1;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
  }

  function choose(value) {
    input.value = value;
    close();
    onChoose(value);
    clearUploadResult();
    updateUploadReadiness();
    input.focus();
  }

  function render() {
    const matches = autocompleteMatches(valuesProvider(), input.value);
    menu.innerHTML = "";
    activeIndex = -1;

    if (matches.length === 0) {
      const empty = document.createElement("div");
      empty.className = "autocomplete-empty";
      empty.textContent = "No supported match";
      menu.appendChild(empty);
    }
    else {
      matches.forEach((value, index) => {
        const option = document.createElement("button");
        option.className = "autocomplete-option";
        option.type = "button";
        option.id = `${menu.id}-option-${index}`;
        option.setAttribute("role", "option");
        option.dataset.value = value;
        option.textContent = human(value);
        option.addEventListener("mousedown", (event) => event.preventDefault());
        option.addEventListener("click", () => choose(value));
        menu.appendChild(option);
      });
    }

    menu.hidden = false;
    input.setAttribute("aria-expanded", "true");
  }

  function moveActive(direction) {
    const options = Array.from(
      menu.querySelectorAll(".autocomplete-option")
    );
    if (options.length === 0) {
      return;
    }

    options.forEach((option) => option.classList.remove("is-active"));
    activeIndex = (activeIndex + direction + options.length) % options.length;
    const active = options[activeIndex];
    active.classList.add("is-active");
    active.scrollIntoView({ block: "nearest" });
    input.setAttribute("aria-activedescendant", active.id);
  }

  input.addEventListener("focus", render);
  input.addEventListener("input", render);
  input.addEventListener("blur", () => window.setTimeout(close, 120));
  input.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (menu.hidden) {
        render();
      }
      moveActive(event.key === "ArrowDown" ? 1 : -1);
    }
    else if (event.key === "Enter" && !menu.hidden) {
      const options = Array.from(
        menu.querySelectorAll(".autocomplete-option")
      );
      const selected = options[activeIndex] || options[0];
      if (selected) {
        event.preventDefault();
        choose(selected.dataset.value);
      }
    }
    else if (event.key === "Escape") {
      close();
    }
  });

}


async function loadStaticInferenceOptions() {

  try {
    const response = await fetch(
      INFERENCE_OPTIONS_URL,
      { cache: "no-store" }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const options = await response.json();

    if (Array.isArray(options.parent_categories)) {
      uploadState.parentCategories = options.parent_categories;
    }
    if (Array.isArray(options.part_queries)) {
      uploadState.partQueries = options.part_queries;
    }
    if (options.part_suggestion_groups) {
      uploadState.partSuggestionGroups = options.part_suggestion_groups;
    }
    if (options.parent_part_groups) {
      uploadState.parentPartGroups = options.parent_part_groups;
    }

    populateUploadSuggestions();
    updateUploadReadiness();
  }
  catch (error) {
    console.warn("Static inference options unavailable", error);
  }

}


function populateUploadControls() {

  const modelSelect = $("#upload-model-select");

  modelSelect.innerHTML = "";

  for (const model of state.models) {

    const option = document.createElement("option");
    option.value = model.id.replace(/^final_/, "");
    option.textContent = model.label;
    modelSelect.appendChild(option);

  }

  const preferredModelId = uploadState.selectedModel || "rotation_consistent";
  const preferredModel = state.models.find(
    (model) => model.id.replace(/^final_/, "") === preferredModelId
  );

  if (preferredModel) {
    modelSelect.value = preferredModel.id.replace(/^final_/, "");
  }

  modelSelect.disabled = state.models.length === 0;

  handleUploadObjectInput();
  updateUploadReadiness();

}


function handleUploadObjectInput() {

  const partInput = $("#upload-part-input");
  const suggestions = partSuggestionsForCurrentObject();
  const category = currentUploadCategory();
  const currentPart = partInput.value.trim().toLowerCase();

  partInput.placeholder = category
    ? `Type to search ${human(category.object)} parts`
    : suggestions.length
      ? "Type to search trained parts"
      : "Select an object, then type a part";

  if (
    currentPart
    && !suggestions.some((part) => part.toLowerCase() === currentPart)
  ) {
    partInput.value = "";
  }

  $("#upload-part-menu").hidden = true;
  updateExperimentalNote();
  clearUploadResult();
  updateUploadReadiness();

}


function setUploadStatus(message, type = "") {

  const status = $("#upload-status");
  status.textContent = message;
  status.classList.toggle("is-error", type === "error");
  status.classList.toggle("is-success", type === "success");

}


function clearUploadResult() {

  const image = $("#upload-result-preview");
  image.hidden = true;
  image.removeAttribute("src");
  $("#upload-result-empty").hidden = false;

}


function matchingUploadDimensions() {

  if (!uploadState.imageSize || !uploadState.maskSize) {
    return true;
  }

  return (
    uploadState.imageSize.width === uploadState.maskSize.width
    && uploadState.imageSize.height === uploadState.maskSize.height
  );

}


function updateUploadReadiness() {

  const objectValue = currentUploadObject().toLowerCase();
  const partValue = currentUploadPart().toLowerCase();
  const supportedObject = uploadState.parentCategories.includes(objectValue);
  const supportedPart = partSuggestionsForCurrentObject()
    .some((part) => part.toLowerCase() === partValue);

  updateExperimentalNote();

  const ready = Boolean(
    uploadState.apiOnline
    && uploadState.imageFile
    && uploadState.maskReady
    && currentUploadObject()
    && currentUploadPart()
    && supportedObject
    && supportedPart
    && $("#upload-model-select").value
    && matchingUploadDimensions()
  );

  $("#upload-run-button").disabled = !ready || uploadState.running;
  $("#detect-parent-button").disabled = !(
    uploadState.apiOnline
    && uploadState.automaticParentAvailable
    && uploadState.imageFile
    && !uploadState.running
  );

  if (
    uploadState.imageSize
    && uploadState.maskSize
    && !matchingUploadDimensions()
  ) {
    setUploadStatus(
      `Image is ${uploadState.imageSize.width}×${uploadState.imageSize.height}, `
      + `but mask is ${uploadState.maskSize.width}×${uploadState.maskSize.height}. `
      + "They must match.",
      "error"
    );
  }
  else if (objectValue && !supportedObject) {
    setUploadStatus(
      "Choose an object from the supported COCO category suggestions.",
      "error"
    );
  }
  else if (partValue && !supportedPart) {
    setUploadStatus(
      "Choose a part from the trained part-query suggestions.",
      "error"
    );
  }
  else if (ready && !uploadState.running) {
    setUploadStatus("Inputs ready. Run the trained part-segmentation model.");
  }

}


async function checkInferenceAPI() {

  if (uploadState.apiCheckRunning) {
    return;
  }

  uploadState.apiCheckRunning = true;
  window.clearTimeout(uploadState.apiRetryTimer);

  const badge = $("#upload-api-badge");
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 8000);

  try {

    const response = await fetch(
      inferenceURL("/api/health"),
      {
        cache: "no-store",
        signal: controller.signal,
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const health = await response.json();
    uploadState.apiOnline = health.status === "ok";
    if (typeof health.selected_model === "string" && health.selected_model) {
      uploadState.selectedModel = health.selected_model.replace(/^final_/, "");
      const modelSelect = $("#upload-model-select");
      const selectedModelAvailable = Array.from(modelSelect.options).some(
        (option) => option.value === uploadState.selectedModel
      );
      if (!uploadState.modelSelectionChanged && selectedModelAvailable) {
        modelSelect.value = uploadState.selectedModel;
      }
    }
    uploadState.automaticParentAvailable = Boolean(
      health.automatic_parent_prediction
    );
    if (Array.isArray(health.parent_categories) && health.parent_categories.length) {
      uploadState.parentCategories = health.parent_categories;
    }
    if (Array.isArray(health.part_queries) && health.part_queries.length) {
      uploadState.partQueries = health.part_queries;
    }
    if (health.part_suggestion_groups) {
      uploadState.partSuggestionGroups = health.part_suggestion_groups;
    }
    if (health.parent_part_groups) {
      uploadState.parentPartGroups = health.parent_part_groups;
    }
    populateUploadSuggestions();

    if (!uploadState.apiOnline) {
      throw new Error("API is not ready");
    }

    badge.textContent = uploadState.automaticParentAvailable
      ? "Automatic parent prediction available"
      : "Manual parent mask available";
    badge.classList.remove("is-checking", "is-offline");
    uploadState.apiRetryTimer = null;

  }
  catch (error) {

    console.info("Inference API unavailable", error);
    uploadState.apiOnline = false;
    badge.textContent = "Inference API starting · retrying";
    badge.classList.remove("is-checking");
    badge.classList.add("is-offline");
    setUploadStatus(
      "Waiting for the Python inference API. The page will reconnect automatically."
    );

  }
  finally {

    window.clearTimeout(timeout);
    uploadState.apiCheckRunning = false;
    updateUploadReadiness();

    if (!uploadState.apiOnline) {
      uploadState.apiRetryTimer = window.setTimeout(
        checkInferenceAPI,
        4000
      );
    }

  }

}


function validUploadFile(file) {

  if (!file || !file.type.startsWith("image/")) {
    setUploadStatus("Please select a PNG, JPEG or WebP image.", "error");
    return false;
  }

  if (file.size > 15 * 1024 * 1024) {
    setUploadStatus("The selected file is larger than 15 MB.", "error");
    return false;
  }

  return true;

}


function setMaskEditorEnabled(enabled) {

  for (const selector of [
    "#mask-add-tool",
    "#mask-erase-tool",
    "#mask-brush-size",
    "#mask-reset-button",
  ]) {
    $(selector).disabled = !enabled;
  }

  $("#mask-undo-button").disabled = !enabled || uploadState.undoStack.length === 0;

}


function clearParentMask() {

  const canvas = $("#mask-editor-canvas");
  const context = canvas.getContext("2d");

  context.clearRect(0, 0, canvas.width, canvas.height);
  canvas.hidden = true;
  $("#mask-editor-image").hidden = true;
  $("#upload-mask-empty").hidden = false;

  uploadState.maskFile = null;
  uploadState.maskSize = null;
  uploadState.maskReady = false;
  uploadState.initialMask = null;
  uploadState.undoStack = [];
  setMaskEditorEnabled(false);
  clearUploadResult();
  updateUploadReadiness();

}


function imageDataFromMask(maskImage, width, height) {

  const temporary = document.createElement("canvas");
  temporary.width = width;
  temporary.height = height;
  const temporaryContext = temporary.getContext("2d", { willReadFrequently: true });
  temporaryContext.drawImage(maskImage, 0, 0, width, height);
  const source = temporaryContext.getImageData(0, 0, width, height);
  const overlay = new ImageData(width, height);

  for (let offset = 0; offset < source.data.length; offset += 4) {
    const luminance = (
      source.data[offset]
      + source.data[offset + 1]
      + source.data[offset + 2]
    ) / 3;

    if (luminance >= 128) {
      overlay.data[offset] = 239;
      overlay.data[offset + 1] = 68;
      overlay.data[offset + 2] = 68;
      overlay.data[offset + 3] = 155;
    }
  }

  return overlay;

}


function loadMaskIntoEditor(maskURL) {

  return new Promise((resolve, reject) => {

    if (!uploadState.imageSize || !uploadState.imageURL) {
      reject(new Error("Upload the RGB image before creating its parent mask."));
      return;
    }

    const maskImage = new Image();

    maskImage.onload = () => {

      const size = {
        width: maskImage.naturalWidth,
        height: maskImage.naturalHeight,
      };
      uploadState.maskSize = size;

      if (!matchingUploadDimensions()) {
        updateUploadReadiness();
        reject(new Error("The parent mask dimensions do not match the RGB image."));
        return;
      }

      const canvas = $("#mask-editor-canvas");
      const context = canvas.getContext("2d", { willReadFrequently: true });
      canvas.width = size.width;
      canvas.height = size.height;
      context.putImageData(
        imageDataFromMask(maskImage, size.width, size.height),
        0,
        0
      );

      const editorImage = $("#mask-editor-image");
      editorImage.src = uploadState.imageURL;
      editorImage.hidden = false;
      canvas.hidden = false;
      $("#upload-mask-empty").hidden = true;

      uploadState.maskReady = true;
      uploadState.initialMask = context.getImageData(0, 0, canvas.width, canvas.height);
      uploadState.undoStack = [];
      setMaskEditorEnabled(true);
      clearUploadResult();
      updateUploadReadiness();
      resolve();

    };

    maskImage.onerror = () => reject(new Error("Could not read the parent mask."));
    maskImage.src = maskURL;

  });

}


function setUploadFile(kind, file) {

  if (!validUploadFile(file)) {
    return;
  }

  if (kind === "image") {

    if (uploadState.imageURL) {
      URL.revokeObjectURL(uploadState.imageURL);
    }

    uploadState.imageFile = file;
    uploadState.imageURL = URL.createObjectURL(file);
    uploadState.imageSize = null;
    uploadState.detections = [];
    clearParentMask();

    const preview = $("#upload-image-preview");
    preview.onload = () => {
      uploadState.imageSize = {
        width: preview.naturalWidth,
        height: preview.naturalHeight,
      };

      const previewGrid = $(".upload-preview-grid");
      previewGrid.style.setProperty(
        "--upload-image-ratio",
        `${preview.naturalWidth} / ${preview.naturalHeight}`
      );
      previewGrid.classList.add("has-image");

      updateUploadReadiness();
      setUploadStatus(
        $("#upload-parent-source").value === "automatic"
          ? "Image ready. Predict its parent mask next."
          : "Image ready. Upload its matching parent mask next."
      );
    };
    preview.src = uploadState.imageURL;
    preview.hidden = false;
    $("#upload-image-empty").hidden = true;
    $("#upload-image-name").textContent = file.name;
    $("#image-drop-zone").classList.add("has-file");
    $(".detected-object-field").hidden = true;

  }
  else {

    if (uploadState.maskURL) {
      URL.revokeObjectURL(uploadState.maskURL);
    }

    uploadState.maskFile = file;
    uploadState.maskURL = URL.createObjectURL(file);
    $("#upload-mask-name").textContent = file.name;
    $("#mask-drop-zone").classList.add("has-file");

    loadMaskIntoEditor(uploadState.maskURL)
      .then(() => setUploadStatus("Parent mask ready. Correct it with Add or Erase if needed."))
      .catch((error) => setUploadStatus(error.message, "error"));

  }

  clearUploadResult();
  updateUploadReadiness();

}


function messageFromAPI(payload, response) {

  if (payload?.detail) {
    return typeof payload.detail === "string"
      ? payload.detail
      : JSON.stringify(payload.detail);
  }

  if (response.status === 404) {
    return "Inference API is not connected. Run the Python inference server, not the static-only server.";
  }

  return `Prediction failed with HTTP ${response.status}.`;

}


async function selectParentDetection(index) {

  const detection = uploadState.detections[index];
  if (!detection) {
    return;
  }

  $("#upload-object-input").value = detection.category;
  $("#upload-part-input").value = "";
  handleUploadObjectInput();

  try {
    await loadMaskIntoEditor(detection.mask_data_url);
    setUploadStatus(
      `${human(detection.category)} detected with `
      + `${Math.round(detection.score * 100)}% confidence. `
      + "Correct the red parent mask if needed."
    );
  }
  catch (error) {
    setUploadStatus(error.message, "error");
  }

}


async function detectParentMask() {

  if (!uploadState.imageFile || $("#detect-parent-button").disabled) {
    return;
  }

  uploadState.running = true;
  updateUploadReadiness();
  clearParentMask();
  $("#automatic-parent-panel").classList.add("is-detecting");
  $("#detect-parent-button").textContent = "Detecting...";
  setUploadStatus(
    "Finding parent objects. The first request downloads and loads the detector..."
  );

  const form = new FormData();
  form.append("image", uploadState.imageFile);

  try {

    const response = await fetch(
      inferenceURL("/api/parent/predict"),
      {
        method: "POST",
        body: form,
      }
    );
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(messageFromAPI(payload, response));
    }

    uploadState.detections = payload.detections || [];

    if (uploadState.detections.length === 0) {
      $("#upload-parent-source").value = "manual";
      handleParentSourceChange();
      throw new Error(
        "No COCO object was detected above the confidence threshold. Upload a parent mask manually instead."
      );
    }

    const detectionField = $(".detected-object-field");
    const detectionSelect = $("#detected-object-select");
    detectionSelect.innerHTML = "";

    uploadState.detections.forEach((detection, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = (
        `${human(detection.category)} · ${Math.round(detection.score * 100)}%`
      );
      detectionSelect.appendChild(option);
    });

    detectionField.hidden = false;
    await selectParentDetection(0);

  }
  catch (error) {
    console.error(error);
    setUploadStatus(error.message || "Parent prediction failed.", "error");
  }
  finally {
    uploadState.running = false;
    $("#automatic-parent-panel").classList.remove("is-detecting");
    $("#detect-parent-button").textContent = "Predict parent mask";
    updateUploadReadiness();
  }

}


function canvasPoint(event) {

  const canvas = $("#mask-editor-canvas");
  const bounds = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) * canvas.width / bounds.width,
    y: (event.clientY - bounds.top) * canvas.height / bounds.height,
  };

}


function saveMaskUndoState() {

  const canvas = $("#mask-editor-canvas");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  uploadState.undoStack.push(
    context.getImageData(0, 0, canvas.width, canvas.height)
  );

  if (uploadState.undoStack.length > 6) {
    uploadState.undoStack.shift();
  }

  setMaskEditorEnabled(true);

}


function paintMask(from, to) {

  const canvas = $("#mask-editor-canvas");
  const context = canvas.getContext("2d");
  const bounds = canvas.getBoundingClientRect();
  const brushSize = Number($("#mask-brush-size").value);
  const nativeSize = brushSize * canvas.width / bounds.width;

  context.save();
  context.globalCompositeOperation = uploadState.brushMode === "erase"
    ? "destination-out"
    : "source-over";
  context.strokeStyle = "rgba(239, 68, 68, 0.61)";
  context.fillStyle = "rgba(239, 68, 68, 0.61)";
  context.lineWidth = nativeSize;
  context.lineCap = "round";
  context.lineJoin = "round";

  context.beginPath();
  context.moveTo(from.x, from.y);
  context.lineTo(to.x, to.y);
  context.stroke();

  if (from.x === to.x && from.y === to.y) {
    context.beginPath();
    context.arc(to.x, to.y, nativeSize / 2, 0, Math.PI * 2);
    context.fill();
  }

  context.restore();

}


function beginMaskStroke(event) {

  if (!uploadState.maskReady || event.button !== 0) {
    return;
  }

  event.preventDefault();
  const canvas = $("#mask-editor-canvas");
  canvas.setPointerCapture(event.pointerId);
  saveMaskUndoState();
  uploadState.drawing = true;
  uploadState.lastPoint = canvasPoint(event);
  paintMask(uploadState.lastPoint, uploadState.lastPoint);
  clearUploadResult();

}


function continueMaskStroke(event) {

  if (!uploadState.drawing) {
    return;
  }

  event.preventDefault();
  const point = canvasPoint(event);
  paintMask(uploadState.lastPoint, point);
  uploadState.lastPoint = point;

}


function finishMaskStroke() {

  if (!uploadState.drawing) {
    return;
  }

  uploadState.drawing = false;
  uploadState.lastPoint = null;
  setUploadStatus("Parent mask corrected. It is ready for part segmentation.");
  updateUploadReadiness();

}


function setMaskBrushMode(mode) {

  uploadState.brushMode = mode;

  for (const candidate of ["add", "erase"]) {
    const button = $(`#mask-${candidate}-tool`);
    const selected = candidate === mode;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", selected ? "true" : "false");
  }

}


function undoMaskEdit() {

  const previous = uploadState.undoStack.pop();
  if (!previous) {
    return;
  }

  $("#mask-editor-canvas").getContext("2d").putImageData(previous, 0, 0);
  setMaskEditorEnabled(true);
  clearUploadResult();
  setUploadStatus("Last mask edit undone.");

}


function resetMaskEditor() {

  if (!uploadState.initialMask) {
    return;
  }

  saveMaskUndoState();
  $("#mask-editor-canvas").getContext("2d").putImageData(
    uploadState.initialMask,
    0,
    0
  );
  clearUploadResult();
  setUploadStatus("Parent mask reset to its initial prediction.");

}


function editedMaskBlob() {

  const source = $("#mask-editor-canvas");
  const sourceContext = source.getContext("2d", { willReadFrequently: true });
  const sourceData = sourceContext.getImageData(0, 0, source.width, source.height);
  const output = document.createElement("canvas");
  output.width = source.width;
  output.height = source.height;
  const outputContext = output.getContext("2d");
  const mask = outputContext.createImageData(output.width, output.height);

  for (let offset = 0; offset < sourceData.data.length; offset += 4) {
    const selected = sourceData.data[offset + 3] > 20 ? 255 : 0;
    mask.data[offset] = selected;
    mask.data[offset + 1] = selected;
    mask.data[offset + 2] = selected;
    mask.data[offset + 3] = 255;
  }

  outputContext.putImageData(mask, 0, 0);

  return new Promise((resolve, reject) => {
    output.toBlob(
      (blob) => blob ? resolve(blob) : reject(new Error("Could not export the corrected mask.")),
      "image/png"
    );
  });

}


async function runUploadedInference() {

  if ($("#upload-run-button").disabled) {
    return;
  }

  uploadState.running = true;
  updateUploadReadiness();
  clearUploadResult();
  setUploadStatus(
    "Loading the model and predicting. The first request can take longer..."
  );

  try {

    const maskBlob = await editedMaskBlob();
    const form = new FormData();
    form.append("image", uploadState.imageFile);
    form.append("parent_mask", maskBlob, "corrected-parent-mask.png");
    form.append("category", currentUploadObject());
    form.append("part", currentUploadPart());
    form.append("model", $("#upload-model-select").value);

    const response = await fetch(
      inferenceURL("/api/predict"),
      {
        method: "POST",
        body: form,
      }
    );

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : null;

    if (!response.ok) {
      throw new Error(messageFromAPI(payload, response));
    }

    if (!payload?.overlay_data_url) {
      throw new Error("The inference API returned no prediction image.");
    }

    const result = $("#upload-result-preview");
    result.src = payload.overlay_data_url;
    result.hidden = false;
    $("#upload-result-empty").hidden = true;

    const experimental = payload.evaluation_mode === "open_vocabulary_experimental";
    setUploadStatus(
      `${human(payload.category)} / ${human(payload.part)} predicted with `
      + `${human(payload.model)} at threshold ${payload.threshold}.`
      + (experimental ? " Experimental open-vocabulary result." : ""),
      "success"
    );
    showToast("Part prediction complete");

  }
  catch (error) {

    console.error(error);
    setUploadStatus(error.message || "Prediction failed.", "error");

  }
  finally {

    uploadState.running = false;
    updateUploadReadiness();

  }

}


function setupDropZone(zoneSelector, kind) {

  const zone = $(zoneSelector);

  for (const eventName of ["dragenter", "dragover"]) {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("is-dragging");
    });
  }

  for (const eventName of ["dragleave", "drop"]) {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("is-dragging");
    });
  }

  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      setUploadFile(kind, file);
    }
  });

}


function handleParentSourceChange() {

  const automatic = $("#upload-parent-source").value === "automatic";
  $("#automatic-parent-panel").hidden = !automatic;
  $("#mask-drop-zone").hidden = automatic;
  $(".detected-object-field").hidden = true;
  clearParentMask();

  setUploadStatus(
    automatic
      ? uploadState.imageFile
        ? "Image ready. Predict its parent mask next."
        : "Upload an RGB image to predict its parent mask."
      : uploadState.imageFile
        ? "Upload a matching white-on-black parent mask."
        : "Upload an RGB image and a matching parent mask."
  );

}


function setupUploadLab() {

  loadStaticInferenceOptions();
  checkInferenceAPI();

  setupAutocomplete(
    "#upload-object-input",
    "#upload-object-menu",
    () => uploadState.parentCategories,
    handleUploadObjectInput
  );
  setupAutocomplete(
    "#upload-part-input",
    "#upload-part-menu",
    partSuggestionsForCurrentObject,
    updateExperimentalNote
  );

  $("#upload-parent-source").addEventListener(
    "change",
    handleParentSourceChange
  );

  $("#upload-image-input").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadFile("image", file);
    }
  });

  $("#upload-mask-input").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadFile("mask", file);
    }
  });

  $("#upload-object-input").addEventListener(
    "input",
    handleUploadObjectInput
  );

  $("#upload-part-input").addEventListener("input", () => {
    updateExperimentalNote();
    clearUploadResult();
    updateUploadReadiness();
  });

  $("#upload-model-select").addEventListener("change", () => {
    uploadState.modelSelectionChanged = true;
    clearUploadResult();
    updateUploadReadiness();
  });

  $("#upload-run-button").addEventListener(
    "click",
    runUploadedInference
  );

  $("#detect-parent-button").addEventListener(
    "click",
    detectParentMask
  );

  $("#detected-object-select").addEventListener("change", (event) => {
    selectParentDetection(Number(event.target.value));
  });

  $("#mask-add-tool").addEventListener("click", () => setMaskBrushMode("add"));
  $("#mask-erase-tool").addEventListener("click", () => setMaskBrushMode("erase"));
  $("#mask-undo-button").addEventListener("click", undoMaskEdit);
  $("#mask-reset-button").addEventListener("click", resetMaskEditor);

  const maskCanvas = $("#mask-editor-canvas");
  maskCanvas.addEventListener("pointerdown", beginMaskStroke);
  maskCanvas.addEventListener("pointermove", continueMaskStroke);
  maskCanvas.addEventListener("pointerup", finishMaskStroke);
  maskCanvas.addEventListener("pointercancel", finishMaskStroke);

  setupDropZone("#image-drop-zone", "image");
  setupDropZone("#mask-drop-zone", "mask");

  handleParentSourceChange();

}


function setDemoMode(mode) {

  const showingUpload = mode === "upload";
  $("#upload-demo-panel").hidden = !showingUpload;
  $("#catalogue-demo-panel").hidden = showingUpload;

  for (const button of $$(".demo-mode-tab")) {
    const selected = button.dataset.demoMode === mode;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", selected ? "true" : "false");
    button.tabIndex = selected ? 0 : -1;
  }

}


function setupDemoModeSwitch() {

  for (const button of $$(".demo-mode-tab")) {
    button.addEventListener("click", () => {
      setDemoMode(button.dataset.demoMode);
    });
  }

  setDemoMode("catalogue");

}


setupDemoModeSwitch();
setupUploadLab();
setupEnhancedExperience();
loadCatalogue();
