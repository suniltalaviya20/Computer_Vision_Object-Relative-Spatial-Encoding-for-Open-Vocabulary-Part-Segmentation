// ============================================================
// CONSTANTS
// ============================================================

const CATALOGUE_URL =
  "data/catalogue.json";


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

  conditionId: "clean",

  comparing: false,

  linkedViews: true,

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
// POPULATE OBJECT SELECT
// ============================================================

function populateObjects() {

  const select =
    $("#object-select");


  const objects = [
    ...new Set(
      state.samples.map(
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

      state.samples

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
      sample.object === objectName
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

  const query =
    $(
      `[data-query="${viewerId}"]`
    );

  const checkpoint =
    $(
      `[data-checkpoint="${viewerId}"]`
    );


  if (!result) {

    iou.textContent =
      "—";

    dice.textContent =
      "—";

    leakage.textContent =
      "—";

    query.textContent =
      "—";

    checkpoint.textContent =
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


  query.textContent =
    state.sample
      ? `Query: "${state.sample.query}"`
      : "—";


  checkpoint.textContent =
    result.checkpoint_id
    ? result.checkpoint_id
        .split("/")
        .slice(-2)
        .join("/")
    : "—";

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

  const id =
    $("#example-id");


  if (!state.sample) {

    name.textContent =
      "—";

    id.textContent =
      "No sample selected";

    return;

  }


  name.textContent =
    `${human(
      state.sample.object
    )} / ${human(
      state.sample.part
    )}`;


  id.textContent =
    state.sample.id;

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

  const linked =
    $("#link-views-container");

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


  linked.hidden =
    !state.comparing;


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
    && state.linkedViews
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
    && state.linkedViews
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
    && state.linkedViews
  ) {

    state.viewers.b.viewId =
      state.viewers.a.viewId;

  }


  render();

}



// ============================================================
// LINK VIEWS
// ============================================================

function handleLinkViews(
  event
) {

  state.linkedViews =
    event.target.checked;


  if (
    state.linkedViews
  ) {

    const view =
      state.viewers.a.viewId;


    if (
      hasView(
        state.viewers.b.modelId,
        view
      )
    ) {

      state.viewers.b.viewId =
        view;

    }

  }


  render();

}



// ============================================================
// INITIAL SAMPLE
// ============================================================

function chooseInitialSample() {

  if (
    state.samples.length === 0
  ) {
    return;
  }


  state.sample =
    state.samples[0];


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


    populateObjects();

    populateModels();

    chooseInitialSample();


    $("#status-badge")
      .classList.add(
        "connected"
      );


    $("#status-text")
      .textContent =
        "Demo ready";


    render();

  }

  catch (
    error
  ) {

    console.error(
      error
    );


    $("#status-text")
      .textContent =
        "Data unavailable";


    $("#setup-notice")
      .hidden =
        false;

  }

}



// ============================================================
// EVENTS
// ============================================================

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


$("#link-views")
  .addEventListener(
    "change",
    handleLinkViews
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

loadCatalogue();