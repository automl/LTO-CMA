# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gps.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='gps.proto',
  package='gps',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\tgps.proto\x12\x03gps\"x\n\x06Sample\x12\t\n\x01T\x18\x01 \x01(\r\x12\n\n\x02\x64X\x18\x02 \x01(\r\x12\n\n\x02\x64U\x18\x03 \x01(\r\x12\n\n\x02\x64O\x18\x04 \x01(\r\x12\r\n\x01X\x18\x05 \x03(\x02\x42\x02\x10\x01\x12\r\n\x01U\x18\x06 \x03(\x02\x42\x02\x10\x01\x12\x0f\n\x03obs\x18\x07 \x03(\x02\x42\x02\x10\x01\x12\x10\n\x04meta\x18\x08 \x03(\x02\x42\x02\x10\x01*\xc7\x01\n\nSampleType\x12\n\n\x06\x41\x43TION\x10\x00\x12\x0b\n\x07\x43UR_LOC\x10\x01\x12\x17\n\x13PAST_OBJ_VAL_DELTAS\x10\x02\x12\r\n\tCUR_SIGMA\x10\x03\x12\n\n\x06\x43UR_PS\x10\x04\x12\x13\n\x0fPAST_LOC_DELTAS\x10\x05\x12\x0e\n\nPAST_SIGMA\x10\x06\x12\x0e\n\nMULTIMODAL\x10\x07\x12\x12\n\x0eGAUSSIAN_NOISE\x10\x08\x12\x10\n\x0c\x43\x41UCHY_NOISE\x10\t\x12\x11\n\rUNIFORM_NOISE\x10\n')
)

_SAMPLETYPE = _descriptor.EnumDescriptor(
  name='SampleType',
  full_name='gps.SampleType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ACTION', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUR_LOC', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PAST_OBJ_VAL_DELTAS', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUR_SIGMA', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUR_PS', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PAST_LOC_DELTAS', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PAST_SIGMA', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MULTIMODAL', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GAUSSIAN_NOISE', index=8, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CAUCHY_NOISE', index=9, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNIFORM_NOISE', index=10, number=10,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=141,
  serialized_end=340,
)
_sym_db.RegisterEnumDescriptor(_SAMPLETYPE)

SampleType = enum_type_wrapper.EnumTypeWrapper(_SAMPLETYPE)
ACTION = 0
CUR_LOC = 1
PAST_OBJ_VAL_DELTAS = 2
CUR_SIGMA = 3
CUR_PS = 4
PAST_LOC_DELTAS = 5
PAST_SIGMA = 6
MULTIMODAL = 7
GAUSSIAN_NOISE = 8
CAUCHY_NOISE = 9
UNIFORM_NOISE = 10



_SAMPLE = _descriptor.Descriptor(
  name='Sample',
  full_name='gps.Sample',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='T', full_name='gps.Sample.T', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dX', full_name='gps.Sample.dX', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dU', full_name='gps.Sample.dU', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dO', full_name='gps.Sample.dO', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='X', full_name='gps.Sample.X', index=4,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='U', full_name='gps.Sample.U', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='obs', full_name='gps.Sample.obs', index=6,
      number=7, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='meta', full_name='gps.Sample.meta', index=7,
      number=8, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=18,
  serialized_end=138,
)

DESCRIPTOR.message_types_by_name['Sample'] = _SAMPLE
DESCRIPTOR.enum_types_by_name['SampleType'] = _SAMPLETYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Sample = _reflection.GeneratedProtocolMessageType('Sample', (_message.Message,), dict(
  DESCRIPTOR = _SAMPLE,
  __module__ = 'gps_pb2'
  # @@protoc_insertion_point(class_scope:gps.Sample)
  ))
_sym_db.RegisterMessage(Sample)


_SAMPLE.fields_by_name['X']._options = None
_SAMPLE.fields_by_name['U']._options = None
_SAMPLE.fields_by_name['obs']._options = None
_SAMPLE.fields_by_name['meta']._options = None
# @@protoc_insertion_point(module_scope)
