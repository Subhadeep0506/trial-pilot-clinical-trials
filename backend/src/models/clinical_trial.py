import json
import re
from datetime import datetime

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.documents import Document

_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m", "%Y")
_SENTINEL_STRINGS = {"", "n/a", "na", "none"}
_AGE_RE = re.compile(r"^\s*([\d.]+)\s+(\w+)\s*$")
_AGE_UNIT_TO_YEARS = {
    "year": 1.0,
    "years": 1.0,
    "month": 1 / 12,
    "months": 1 / 12,
    "week": 1 / 52,
    "weeks": 1 / 52,
    "day": 1 / 365,
    "days": 1 / 365,
    "hour": 1 / (365 * 24),
    "hours": 1 / (365 * 24),
}


def _parse_ctgov_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _parse_age_to_years(age_str: str | None) -> float | None:
    if not age_str:
        return None
    match = _AGE_RE.match(age_str)
    if not match:
        return None
    value, unit = match.groups()
    factor = _AGE_UNIT_TO_YEARS.get(unit.lower())
    if factor is None:
        return None
    try:
        return round(float(value) * factor, 4)
    except ValueError:
        return None


def _clean_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _SENTINEL_STRINGS else text


class _CTGovBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class OrgStudyIdInfo(_CTGovBaseModel):
    id: str | None = None
    link: str | None = None
    type: str | None = None


class ExpandedAccessInfo(_CTGovBaseModel):
    has_expanded_access: bool | None = None
    nct_id: str | None = None
    status_for_nct_id: str | None = None


class DateStruct(_CTGovBaseModel):
    date: str | None = None
    type: str | None = None
    parsed_date: datetime | None = None


class MaskingInfo(_CTGovBaseModel):
    masking: str | None = None
    masking_description: str | None = None
    who_masked: list[str] = Field(default_factory=list)


class DesignInfo(_CTGovBaseModel):
    allocation: str | None = None
    intervention_model: str | None = None
    intervention_model_description: str | None = None
    masking_info: MaskingInfo | None = None
    observational_model: str | None = None
    primary_purpose: str | None = None
    time_perspective: str | None = None


class GeoPoint(_CTGovBaseModel):
    lat: float | None = None
    lon: float | None = None


class Location(_CTGovBaseModel):
    facility: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    geo_point: GeoPoint | None = None


class SponsorInfo(_CTGovBaseModel):
    name: str | None = None
    org_class: str | None = None


class ArmGroup(_CTGovBaseModel):
    label: str | None = None
    type: str | None = None
    description: str | None = None
    intervention_names: list[str] = Field(default_factory=list)


class Intervention(_CTGovBaseModel):
    type: str | None = None
    name: str | None = None
    description: str | None = None
    arm_group_labels: list[str] = Field(default_factory=list)
    other_names: list[str] = Field(default_factory=list)


class Outcome(_CTGovBaseModel):
    measure: str | None = None
    description: str | None = None
    time_frame: str | None = None


class Outcomes(_CTGovBaseModel):
    primary: list[Outcome] = Field(default_factory=list)
    secondary: list[Outcome] = Field(default_factory=list)
    other: list[Outcome] = Field(default_factory=list)


class OverallOfficial(_CTGovBaseModel):
    name: str | None = None
    affiliation: str | None = None
    role: str | None = None


class MeshTerm(_CTGovBaseModel):
    id: str | None = None
    term: str | None = None


class BrowseBranch(_CTGovBaseModel):
    abbrev: str | None = None
    name: str | None = None


class BrowseLeaf(_CTGovBaseModel):
    id: str | None = None
    name: str | None = None
    as_found: str | None = None
    relevance: str | None = None


class BrowseModule(_CTGovBaseModel):
    meshes: list[MeshTerm] = Field(default_factory=list)
    ancestors: list[MeshTerm] = Field(default_factory=list)
    browse_branches: list[BrowseBranch] = Field(default_factory=list)
    browse_leaves: list[BrowseLeaf] = Field(default_factory=list)


class MeshTerms(_CTGovBaseModel):
    conditions: list[MeshTerm] = Field(default_factory=list)
    interventions: list[MeshTerm] = Field(default_factory=list)


class MiscInfoModule(_CTGovBaseModel):
    version_holder: str | None = None


class ClinicalTrial(_CTGovBaseModel):
    nct_id: str
    brief_title: str | None = None
    official_title: str | None = None
    acronym: str | None = None
    study_type: str | None = None
    overall_status: str | None = None

    study_first_submit_date: datetime | None = None
    study_first_submit_qc_date: datetime | None = None
    study_first_post_date: DateStruct | None = None
    last_update_submit_date: datetime | None = None
    last_update_post_date: DateStruct | None = None
    start_date: DateStruct | None = None
    primary_completion_date: DateStruct | None = None
    completion_date: DateStruct | None = None

    phases: list[str] = Field(default_factory=list)
    enrollment_count: int | None = None
    enrollment_type: str | None = None
    minimum_age: float | None = None
    maximum_age: float | None = None
    sex: str | None = None
    healthy_volunteers: bool | None = None
    std_ages: list[str] = Field(default_factory=list)
    study_population: str | None = None
    sampling_method: str | None = None

    brief_summary: str | None = None
    detailed_description: str | None = None
    eligibility_criteria: str | None = None

    lead_sponsor_name: str | None = None
    lead_sponsor_class: str | None = None
    org_study_id_info: OrgStudyIdInfo | None = None
    collaborators: list[SponsorInfo] = Field(default_factory=list)

    why_stopped: str | None = None
    expanded_access_info: ExpandedAccessInfo | None = None
    oversight_has_dmc: bool | None = None

    design_info: DesignInfo | None = None
    conditions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    arm_groups: list[ArmGroup] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    overall_officials: list[OverallOfficial] = Field(default_factory=list)
    outcomes: Outcomes | None = None
    study_references: str | None = None

    misc_info_module: MiscInfoModule | None = None
    condition_browse_module: BrowseModule | None = None
    intervention_browse_module: BrowseModule | None = None
    mesh_terms: MeshTerms | None = None

    has_results: bool = False
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def from_raw_api_response(cls, raw: dict) -> "ClinicalTrial":
        protocol = raw.get("protocolSection") or {}
        derived = raw.get("derivedSection") or {}

        identification = protocol.get("identificationModule") or {}
        status = protocol.get("statusModule") or {}
        sponsor_collab = protocol.get("sponsorCollaboratorsModule") or {}
        oversight = protocol.get("oversightModule") or {}
        description = protocol.get("descriptionModule") or {}
        conditions_mod = protocol.get("conditionsModule") or {}
        design = protocol.get("designModule") or {}
        arms_interventions = protocol.get("armsInterventionsModule") or {}
        outcomes_mod = protocol.get("outcomesModule") or {}
        eligibility = protocol.get("eligibilityModule") or {}
        contacts_locations = protocol.get("contactsLocationsModule") or {}
        references = protocol.get("referencesModule") or {}

        misc_info = derived.get("miscInfoModule") or {}
        condition_browse = derived.get("conditionBrowseModule") or {}
        intervention_browse = derived.get("interventionBrowseModule") or {}

        org_study_id_info_raw = identification.get("orgStudyIdInfo")
        org_study_id_info = (
            OrgStudyIdInfo(
                id=_clean_str(org_study_id_info_raw.get("id")),
                link=_clean_str(org_study_id_info_raw.get("link")),
                type=_clean_str(org_study_id_info_raw.get("type")),
            )
            if org_study_id_info_raw
            else None
        )

        lead_sponsor = sponsor_collab.get("leadSponsor") or {}
        collaborators = [
            SponsorInfo(
                name=_clean_str(c.get("name")), org_class=_clean_str(c.get("class"))
            )
            for c in sponsor_collab.get("collaborators") or []
        ]

        expanded_access_raw = status.get("expandedAccessInfo")
        expanded_access_info = (
            ExpandedAccessInfo(
                has_expanded_access=expanded_access_raw.get("hasExpandedAccess"),
                nct_id=_clean_str(expanded_access_raw.get("nctId")),
                status_for_nct_id=_clean_str(expanded_access_raw.get("statusForNctId")),
            )
            if expanded_access_raw
            else None
        )

        def _date_struct(raw_struct):
            if not raw_struct:
                return None
            date_str = raw_struct.get("date")
            return DateStruct(
                date=date_str,
                type=_clean_str(raw_struct.get("type")),
                parsed_date=_parse_ctgov_date(date_str),
            )

        design_info_raw = design.get("designInfo") or {}
        masking_info_raw = design_info_raw.get("maskingInfo")
        masking_info = (
            MaskingInfo(
                masking=_clean_str(masking_info_raw.get("masking")),
                masking_description=_clean_str(
                    masking_info_raw.get("maskingDescription")
                ),
                who_masked=masking_info_raw.get("whoMasked") or [],
            )
            if masking_info_raw
            else None
        )
        design_info = (
            DesignInfo(
                allocation=_clean_str(design_info_raw.get("allocation")),
                intervention_model=_clean_str(design_info_raw.get("interventionModel")),
                intervention_model_description=_clean_str(
                    design_info_raw.get("interventionModelDescription")
                ),
                masking_info=masking_info,
                observational_model=_clean_str(
                    design_info_raw.get("observationalModel")
                ),
                primary_purpose=_clean_str(design_info_raw.get("primaryPurpose")),
                time_perspective=_clean_str(design_info_raw.get("timePerspective")),
            )
            if design_info_raw
            else None
        )

        locations = [
            Location(
                facility=_clean_str(loc.get("facility")),
                city=_clean_str(loc.get("city")),
                state=_clean_str(loc.get("state")),
                country=_clean_str(loc.get("country")),
                geo_point=(
                    GeoPoint(
                        lat=loc["geoPoint"].get("lat"), lon=loc["geoPoint"].get("lon")
                    )
                    if loc.get("geoPoint")
                    else None
                ),
            )
            for loc in contacts_locations.get("locations") or []
        ]

        arm_groups = [
            ArmGroup(
                label=_clean_str(a.get("label")),
                type=_clean_str(a.get("type")),
                description=_clean_str(a.get("description")),
                intervention_names=a.get("interventionNames") or [],
            )
            for a in arms_interventions.get("armGroups") or []
        ]

        interventions = [
            Intervention(
                type=_clean_str(i.get("type")),
                name=_clean_str(i.get("name")),
                description=_clean_str(i.get("description")),
                arm_group_labels=i.get("armGroupLabels") or [],
                other_names=i.get("otherNames") or [],
            )
            for i in arms_interventions.get("interventions") or []
        ]

        def _outcome_list(raw_list):
            return [
                Outcome(
                    measure=_clean_str(o.get("measure")),
                    description=_clean_str(o.get("description")),
                    time_frame=_clean_str(o.get("timeFrame")),
                )
                for o in raw_list or []
            ]

        outcomes = Outcomes(
            primary=_outcome_list(outcomes_mod.get("primaryOutcomes")),
            secondary=_outcome_list(outcomes_mod.get("secondaryOutcomes")),
            other=_outcome_list(outcomes_mod.get("otherOutcomes")),
        )

        overall_officials = [
            OverallOfficial(
                name=_clean_str(o.get("name")),
                affiliation=_clean_str(o.get("affiliation")),
                role=_clean_str(o.get("role")),
            )
            for o in contacts_locations.get("overallOfficials") or []
        ]

        def _mesh_list(raw_list):
            return [
                MeshTerm(id=m.get("id"), term=m.get("term")) for m in raw_list or []
            ]

        def _browse_module(raw_module):
            if not raw_module:
                return None
            return BrowseModule(
                meshes=_mesh_list(raw_module.get("meshes")),
                ancestors=_mesh_list(raw_module.get("ancestors")),
                browse_branches=[
                    BrowseBranch(abbrev=b.get("abbrev"), name=b.get("name"))
                    for b in raw_module.get("browseBranches") or []
                ],
                browse_leaves=[
                    BrowseLeaf(
                        id=leaf.get("id"),
                        name=leaf.get("name"),
                        as_found=leaf.get("asFound"),
                        relevance=leaf.get("relevance"),
                    )
                    for leaf in raw_module.get("browseLeaves") or []
                ],
            )

        condition_browse_module = _browse_module(condition_browse)
        intervention_browse_module = _browse_module(intervention_browse)
        mesh_terms = MeshTerms(
            conditions=(
                condition_browse_module.meshes if condition_browse_module else []
            ),
            interventions=(
                intervention_browse_module.meshes if intervention_browse_module else []
            ),
        )

        misc_info_module = (
            MiscInfoModule(version_holder=_clean_str(misc_info.get("versionHolder")))
            if misc_info
            else None
        )

        enrollment_info = design.get("enrollmentInfo") or {}
        study_references_raw = references.get("references")

        return cls(
            nct_id=identification.get("nctId"),
            brief_title=_clean_str(identification.get("briefTitle")),
            official_title=_clean_str(identification.get("officialTitle")),
            acronym=_clean_str(identification.get("acronym")),
            study_type=_clean_str(design.get("studyType")),
            overall_status=_clean_str(status.get("overallStatus")),
            study_first_submit_date=_parse_ctgov_date(
                status.get("studyFirstSubmitDate")
            ),
            study_first_submit_qc_date=_parse_ctgov_date(
                status.get("studyFirstSubmitQcDate")
            ),
            study_first_post_date=_date_struct(status.get("studyFirstPostDateStruct")),
            last_update_submit_date=_parse_ctgov_date(
                status.get("lastUpdateSubmitDate")
            ),
            last_update_post_date=_date_struct(status.get("lastUpdatePostDateStruct")),
            start_date=_date_struct(status.get("startDateStruct")),
            primary_completion_date=_date_struct(
                status.get("primaryCompletionDateStruct")
            ),
            completion_date=_date_struct(status.get("completionDateStruct")),
            phases=design.get("phases") or [],
            enrollment_count=enrollment_info.get("count"),
            enrollment_type=_clean_str(enrollment_info.get("type")),
            minimum_age=_parse_age_to_years(eligibility.get("minimumAge")),
            maximum_age=_parse_age_to_years(eligibility.get("maximumAge")),
            sex=_clean_str(eligibility.get("sex")),
            healthy_volunteers=eligibility.get("healthyVolunteers"),
            std_ages=eligibility.get("stdAges") or [],
            study_population=_clean_str(eligibility.get("studyPopulation")),
            sampling_method=_clean_str(eligibility.get("samplingMethod")),
            brief_summary=_clean_str(description.get("briefSummary")),
            detailed_description=_clean_str(description.get("detailedDescription")),
            eligibility_criteria=_clean_str(eligibility.get("eligibilityCriteria")),
            lead_sponsor_name=_clean_str(lead_sponsor.get("name")),
            lead_sponsor_class=_clean_str(lead_sponsor.get("class")),
            org_study_id_info=org_study_id_info,
            collaborators=collaborators,
            why_stopped=_clean_str(status.get("whyStopped")),
            expanded_access_info=expanded_access_info,
            oversight_has_dmc=oversight.get("oversightHasDmc"),
            design_info=design_info,
            conditions=conditions_mod.get("conditions") or [],
            keywords=conditions_mod.get("keywords") or [],
            interventions=interventions,
            arm_groups=arm_groups,
            locations=locations,
            overall_officials=overall_officials,
            outcomes=outcomes,
            study_references=(
                json.dumps(study_references_raw) if study_references_raw else None
            ),
            misc_info_module=misc_info_module,
            condition_browse_module=condition_browse_module,
            intervention_browse_module=intervention_browse_module,
            mesh_terms=mesh_terms,
            has_results=bool(raw.get("hasResults", False)),
        )

    def to_embedding_documents(self) -> str:
        """Composite text used to generate this trial's vector embedding."""
        intervention_parts = []
        for iv in self.interventions:
            bits = [b for b in (iv.name, iv.description) if b]
            if bits:
                intervention_parts.append(": ".join(bits))

        mesh_terms: list[str] = []
        if self.mesh_terms:
            mesh_terms = [
                t.term
                for t in self.mesh_terms.conditions + self.mesh_terms.interventions
                if t.term
            ]

        parts: list[str | None] = [
            self.official_title or self.brief_title,
            ", ".join(self.conditions) if self.conditions else None,
            ", ".join(self.keywords) if self.keywords else None,
            ", ".join(self.phases) if self.phases else None,
            "\n".join(intervention_parts) if intervention_parts else None,
            self.brief_summary,
            self.detailed_description,
            self.eligibility_criteria,
            ", ".join(mesh_terms) if mesh_terms else None,
        ]
        return Document(
            page_content="\n".join(p for p in parts if p),
            metadata={
                "nct_id": self.nct_id,
                "brief_title": self.brief_title,
                "official_title": self.official_title,
                "acronym": self.acronym,
                "study_type": self.study_type,
                "overall_status": self.overall_status,
                "study_first_submit_date": (
                    self.study_first_submit_date.isoformat()
                    if self.study_first_submit_date
                    else None
                ),
                "study_first_submit_qc_date": (
                    self.study_first_submit_qc_date.isoformat()
                    if self.study_first_submit_qc_date
                    else None
                ),
                "study_first_post_date": (
                    self.study_first_post_date.parsed_date.isoformat()
                    if self.study_first_post_date
                    and self.study_first_post_date.parsed_date
                    else None
                ),
                "last_update_submit_date": (
                    self.last_update_submit_date.isoformat()
                    if self.last_update_submit_date
                    else None
                ),
                "last_update_post_date": (
                    self.last_update_post_date.parsed_date.isoformat()
                    if self.last_update_post_date
                    and self.last_update_post_date.parsed_date
                    else None
                ),
                "start_date": (
                    self.start_date.parsed_date.isoformat()
                    if self.start_date and self.start_date.parsed_date
                    else None
                ),
                "primary_completion_date": (
                    self.primary_completion_date.parsed_date.isoformat()
                    if self.primary_completion_date
                    and self.primary_completion_date.parsed_date
                    else None
                ),
                "completion_date": (
                    self.completion_date.parsed_date.isoformat()
                    if self.completion_date and self.completion_date.parsed_date
                    else None
                ),
                "phases": self.phases,
                "enrollment_count": self.enrollment_count,
                "enrollment_type": self.enrollment_type,
                "minimum_age": self.minimum_age,
                "maximum_age": self.maximum_age,
                "sex": self.sex,
                "healthy_volunteers": self.healthy_volunteers,
                "std_ages": self.std_ages,
                "study_population": self.study_population,
                "sampling_method": self.sampling_method,
                "lead_sponsor_name": self.lead_sponsor_name,
                "lead_sponsor_class": self.lead_sponsor_class,
                "org_study_id_info": (
                    self.org_study_id_info.model_dump(mode="json")
                    if self.org_study_id_info
                    else None
                ),
                "collaborators": [c.model_dump(mode="json") for c in self.collaborators],
                "why_stopped": self.why_stopped,
                "expanded_access_info": (
                    self.expanded_access_info.model_dump(mode="json")
                    if self.expanded_access_info
                    else None
                ),
                "oversight_has_dmc": self.oversight_has_dmc,
                "design_info": (
                    self.design_info.model_dump(mode="json") if self.design_info else None
                ),
                "interventions": [i.model_dump(mode="json") for i in self.interventions],
                "arm_groups": [a.model_dump(mode="json") for a in self.arm_groups],
                "locations": [loc.model_dump(mode="json") for loc in self.locations],
                "overall_officials": [
                    o.model_dump(mode="json") for o in self.overall_officials
                ],
                "outcomes": self.outcomes.model_dump(mode="json") if self.outcomes else None,
                "study_references": self.study_references,
                "misc_info_module": (
                    self.misc_info_module.model_dump(mode="json")
                    if self.misc_info_module
                    else None
                ),
                "condition_browse_module": (
                    self.condition_browse_module.model_dump(mode="json")
                    if self.condition_browse_module
                    else None
                ),
                "intervention_browse_module": (
                    self.intervention_browse_module.model_dump(mode="json")
                    if self.intervention_browse_module
                    else None
                ),
                "has_results": self.has_results,
            },
        )


_DATE_STRUCT_ARROW = pa.struct(
    [("date", pa.string()), ("type", pa.string()), ("parsed_date", pa.timestamp("us"))]
)
_ORG_STUDY_ID_INFO_ARROW = pa.struct(
    [("id", pa.string()), ("link", pa.string()), ("type", pa.string())]
)
_EXPANDED_ACCESS_INFO_ARROW = pa.struct(
    [
        ("has_expanded_access", pa.bool_()),
        ("nct_id", pa.string()),
        ("status_for_nct_id", pa.string()),
    ]
)
_MASKING_INFO_ARROW = pa.struct(
    [
        ("masking", pa.string()),
        ("masking_description", pa.string()),
        ("who_masked", pa.list_(pa.string())),
    ]
)
_DESIGN_INFO_ARROW = pa.struct(
    [
        ("allocation", pa.string()),
        ("intervention_model", pa.string()),
        ("intervention_model_description", pa.string()),
        ("masking_info", _MASKING_INFO_ARROW),
        ("observational_model", pa.string()),
        ("primary_purpose", pa.string()),
        ("time_perspective", pa.string()),
    ]
)
_GEO_POINT_ARROW = pa.struct([("lat", pa.float64()), ("lon", pa.float64())])
_LOCATION_ARROW = pa.struct(
    [
        ("facility", pa.string()),
        ("city", pa.string()),
        ("state", pa.string()),
        ("country", pa.string()),
        ("geo_point", _GEO_POINT_ARROW),
    ]
)
_SPONSOR_INFO_ARROW = pa.struct([("name", pa.string()), ("org_class", pa.string())])
_ARM_GROUP_ARROW = pa.struct(
    [
        ("label", pa.string()),
        ("type", pa.string()),
        ("description", pa.string()),
        ("intervention_names", pa.list_(pa.string())),
    ]
)
_INTERVENTION_ARROW = pa.struct(
    [
        ("type", pa.string()),
        ("name", pa.string()),
        ("description", pa.string()),
        ("arm_group_labels", pa.list_(pa.string())),
        ("other_names", pa.list_(pa.string())),
    ]
)
_OUTCOME_ARROW = pa.struct(
    [
        ("measure", pa.string()),
        ("description", pa.string()),
        ("time_frame", pa.string()),
    ]
)
_OUTCOMES_ARROW = pa.struct(
    [
        ("primary", pa.list_(_OUTCOME_ARROW)),
        ("secondary", pa.list_(_OUTCOME_ARROW)),
        ("other", pa.list_(_OUTCOME_ARROW)),
    ]
)
_OVERALL_OFFICIAL_ARROW = pa.struct(
    [("name", pa.string()), ("affiliation", pa.string()), ("role", pa.string())]
)
_MESH_TERM_ARROW = pa.struct([("id", pa.string()), ("term", pa.string())])
_BROWSE_BRANCH_ARROW = pa.struct([("abbrev", pa.string()), ("name", pa.string())])
_BROWSE_LEAF_ARROW = pa.struct(
    [
        ("id", pa.string()),
        ("name", pa.string()),
        ("as_found", pa.string()),
        ("relevance", pa.string()),
    ]
)
_BROWSE_MODULE_ARROW = pa.struct(
    [
        ("meshes", pa.list_(_MESH_TERM_ARROW)),
        ("ancestors", pa.list_(_MESH_TERM_ARROW)),
        ("browse_branches", pa.list_(_BROWSE_BRANCH_ARROW)),
        ("browse_leaves", pa.list_(_BROWSE_LEAF_ARROW)),
    ]
)
_MESH_TERMS_ARROW = pa.struct(
    [
        ("conditions", pa.list_(_MESH_TERM_ARROW)),
        ("interventions", pa.list_(_MESH_TERM_ARROW)),
    ]
)
_MISC_INFO_MODULE_ARROW = pa.struct([("version_holder", pa.string())])

CLINICAL_TRIAL_ARROW_SCHEMA = pa.schema(
    [
        ("nct_id", pa.string()),
        ("brief_title", pa.string()),
        ("official_title", pa.string()),
        ("acronym", pa.string()),
        ("study_type", pa.string()),
        ("overall_status", pa.string()),
        ("study_first_submit_date", pa.timestamp("us")),
        ("study_first_submit_qc_date", pa.timestamp("us")),
        ("study_first_post_date", _DATE_STRUCT_ARROW),
        ("last_update_submit_date", pa.timestamp("us")),
        ("last_update_post_date", _DATE_STRUCT_ARROW),
        ("start_date", _DATE_STRUCT_ARROW),
        ("primary_completion_date", _DATE_STRUCT_ARROW),
        ("completion_date", _DATE_STRUCT_ARROW),
        ("phases", pa.list_(pa.string())),
        ("enrollment_count", pa.int64()),
        ("enrollment_type", pa.string()),
        ("minimum_age", pa.float64()),
        ("maximum_age", pa.float64()),
        ("sex", pa.string()),
        ("healthy_volunteers", pa.bool_()),
        ("std_ages", pa.list_(pa.string())),
        ("study_population", pa.string()),
        ("sampling_method", pa.string()),
        ("brief_summary", pa.string()),
        ("detailed_description", pa.string()),
        ("eligibility_criteria", pa.string()),
        ("lead_sponsor_name", pa.string()),
        ("lead_sponsor_class", pa.string()),
        ("org_study_id_info", _ORG_STUDY_ID_INFO_ARROW),
        ("collaborators", pa.list_(_SPONSOR_INFO_ARROW)),
        ("why_stopped", pa.string()),
        ("expanded_access_info", _EXPANDED_ACCESS_INFO_ARROW),
        ("oversight_has_dmc", pa.bool_()),
        ("design_info", _DESIGN_INFO_ARROW),
        ("conditions", pa.list_(pa.string())),
        ("keywords", pa.list_(pa.string())),
        ("interventions", pa.list_(_INTERVENTION_ARROW)),
        ("arm_groups", pa.list_(_ARM_GROUP_ARROW)),
        ("locations", pa.list_(_LOCATION_ARROW)),
        ("overall_officials", pa.list_(_OVERALL_OFFICIAL_ARROW)),
        ("outcomes", _OUTCOMES_ARROW),
        ("study_references", pa.string()),
        ("misc_info_module", _MISC_INFO_MODULE_ARROW),
        ("condition_browse_module", _BROWSE_MODULE_ARROW),
        ("intervention_browse_module", _BROWSE_MODULE_ARROW),
        ("mesh_terms", _MESH_TERMS_ARROW),
        ("has_results", pa.bool_()),
        ("fetched_at", pa.timestamp("us")),
    ]
)
